import math
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config

from transformers.models.mamba.modeling_mamba import (
    selective_scan_fn,
    causal_conv1d_fn,
    mamba_inner_fn,
)
from transformers.models.mamba2.modeling_mamba2 import mamba_chunk_scan_combined, MambaRMSNormGated

is_fast_path_available = all(
    (selective_scan_fn, causal_conv1d_fn, mamba_inner_fn)
)


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.time_step_rank**-0.5 * config.time_step_scale
        if config.time_step_init_scheme == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.time_step_init_scheme == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.intermediate_size) * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp(min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def cuda_kernels_forward(self, hidden_states):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        hidden_states, gate = projected_states.chunk(2, dim=1)
        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        hidden_states = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
        )

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        scan_outputs = selective_scan_fn(
            hidden_states,
            discrete_time_step,
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            self.D.float(),
            gate,
            time_proj_bias,
            delta_softplus=True
        )

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states):
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type
        return self.cuda_kernels_forward(hidden_states)


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.num_heads = self.intermediate_size // self.head_dim

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.time_step_max) - math.log(self.time_step_min)) 
            + math.log(self.time_step_min)
        )
        dt = torch.clamp(dt, min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A_init_range = (1, 16)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.intermediate_size)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def cuda_kernels_forward(self, hidden_states):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log)  # (num_heads) or (intermediate_size, state_size)
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        gate, hidden_states_B_C, time_step = torch.split(
            projected_states,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        time_step = nn.functional.softplus(time_step + self.dt_bias)
        assert self.activation in ["silu", "swish"]
        # 1D Convolution
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
            )
        else:
            hidden_states_B_C = causal_conv1d_fn(
                x=hidden_states_B_C.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)[:, :seq_len]

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        scan_output = mamba_chunk_scan_combined(
            hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            time_step,
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=None,
            **dt_limit_kwargs,
        )
        scan_output = scan_output.view(batch_size, seq_len, -1)
        # Multiply "gate" branch and apply extra normalization layer
        # scan_output = self.norm(scan_output, gate)
        scan_output = self.norm(scan_output) * self.act(gate.to(torch.float32))
        # scan_output = torch.cat([self.norm(scan_output), self.act(gate.to(torch.float32))], dim=-1)
        out = self.out_proj(scan_output)
        return out

    def forward(self, hidden_states):
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type
        return self.cuda_kernels_forward(hidden_states)
    

class Mamba2MixerRMSGated(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.num_heads = self.intermediate_size // self.head_dim

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(self.time_step_max) - math.log(self.time_step_min)) 
            + math.log(self.time_step_min)
        )
        dt = torch.clamp(dt, min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A_init_range = (1, 16)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def cuda_kernels_forward(self, hidden_states):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log)  # (num_heads) or (intermediate_size, state_size)
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        gate, hidden_states_B_C, time_step = torch.split(
            projected_states,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        time_step = nn.functional.softplus(time_step + self.dt_bias)
        assert self.activation in ["silu", "swish"]
        # 1D Convolution
        hidden_states_B_C = self.act(
            self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
        )

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        scan_output = mamba_chunk_scan_combined(
            hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            time_step,
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=None,
            **dt_limit_kwargs,
        )
        scan_output = scan_output.view(batch_size, seq_len, -1)
        # Multiply "gate" branch and apply extra normalization layer
        # scan_output = self.norm(scan_output, gate)
        scan_output = self.norm(scan_output, gate)
        out = self.out_proj(scan_output)
        return out

    def forward(self, hidden_states):
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type
        return self.cuda_kernels_forward(hidden_states)
    

class MambaMixerHuggingFace(nn.Module):
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.time_step_rank**-0.5 * config.time_step_scale
        if config.time_step_init_scheme == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.time_step_init_scheme == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.intermediate_size) * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp(min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def cuda_kernels_forward(self, hidden_states):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            
            scan_outputs = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True
            )

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states)