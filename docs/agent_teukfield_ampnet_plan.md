Teukfield-AmpNet 新分支实现任务书
面向本地 agent / Codex 的完整 PDF
说明：分支创建、最小文件复制、模型结构、损失函数、训练策略、验证标准与里程碑
目标仓库：ljq2088/SolvingTeukolskyEq。建议新分支：research/teukfield-ampnet-20260607。基线分支：
research/high-accuracy-teukolsky-plan-20260528。
角色

指令摘要

本地 agent

在新分支内实现 Teukfield-AmpNet，不要直接改旧分支。

复制策略

只从旧分支复制 spectral 边界参考、物理因子、Chebyshev 工具、诊断 LS；不要复制旧训
练脚本和大结果目录。

主模型

参数 FiLM 调制坐标网络，但主干不是普通 MLP，而是局部 SIREN/
Pirate/
FBPINN neural field。

训练对象

全局 S = R_
in/
(P h2)，并联合训练 amplitude head 输出 B_
inc 与 B_
ref/
B_
inc。

约束闭环

AD Teukolsky residual + spectral boundary consistency + benchmark anchors + amplitude
anchors。

Teukfield-AmpNet agent implementation plan

第1页

0. 一句话任务定义
在新的 Git 分支中实现一个以神经场为主的 Kerr s=-2, l=m=2 radial Teukolsky R_in 求解器。不要再以“谱系数网
络”作为主求解器。旧分支中的谱方法只作为边界附近高精度参考、benchmark、诊断工具和少量 anchor 来源。
主表示为：
S_theta(y,p) = R_in(r;p) / (P(r;p) h2(r;p))
y = 2 r_+/r - 1,
p = (a, log10 omega, k, Omega_H, kappa, lambda, ...)
R_theta(r;p) = P(r;p) h2(r;p) S_theta(y;p)

同时用共享参数编码器给出振幅：
B_inc_theta(p),
rho_theta(p) = B_ref_theta(p) / B_inc_theta(p)
B_ref_theta(p) = rho_theta(p) * B_inc_theta(p)

无穷远边界附近必须联合满足：
P h2 S_theta ~= B_inc_theta A_down u_down_spec + B_ref_theta A_up u_up_spec

视界边界附近必须联合满足：
P h2 S_theta ~= A_in u_in_spec

所有导数尽量从神经场 S_theta 的自动微分得到，发挥 PINN / neural field 的优势；谱微分不再作为主导数路径。

1. 为什么要新分支
旧路线经历了两个极端：纯 PINN 能利用自动微分，但齐次方程尺度自由度强，容易得到低残差但归一化错误
的解；纯谱系数 decoder 稳定，但学习对象变成了谱方法产物，极低频/极高频与振幅恢复仍受谱方法和 LS 条件
数限制。新分支要把二者融合成自然的物理闭环：神经场表示全局解，谱方法只在边界附近提供可信物理锚点。
不要再训练 (a,omega)->Chebyshev coefficients 作为主模型。
不要用 post-hoc multipoint LS 作为最终振幅机制；它只能作为诊断或 pseudo-anchor。
要让 R_in 网络、振幅网络、无穷远谱分支、方程 residual 同时参与训练。
初始先做 moderate patch，验证模型闭环；不要一开始覆盖全参数空间。

2. 分支创建与最小复制规则
本地 agent 应执行以下命令：
git fetch origin
git checkout -b research/teukfield-ampnet-20260607
origin/research/high-accuracy-teukolsky-plan-20260528

然后只复制或保留旧分支中的必要代码。原则：旧谱代码提供 boundary reference 和 diagnostic，不要把旧谱训练
pipeline 整体搬过来。

Teukfield-AmpNet agent implementation plan

第2页

图 1：新分支从旧分支复制最小谱基础设施，避免把旧谱系数训练 pipeline 变成新分支主线。
旧分支文件/
模块

新分支用途

处理方式

utils/
amplitude.py

A_
in, A_
down, A_
up, tortoise, prefactor, reduced equation coefficients 的权威
来源。

复制后重构到 teukfield/
physics/
factors.py 与 teukfield/
physics/
reduced_
equation.py。保留原文件也可，但
新代码应从新模块调用。

teukspec/
core/
chebyshev.py

Chebyshev 节点、quadrature、weak residual 测试函数和
debug 评估。

复制到 teukfield/
physics/
cheb_
tools.py 或保留导入；只作为工具
，不作为主表示。

teukspec/
amplitude/
multipoint_
ls.py

振幅诊断、生成 pseudo-anchor、对比网络 amplitude
head。

复制/
保留，但标记为 diagnostics only。
不要作为最终 amplitude solver。

teukspec/
decoders/
param_
cheb_
decoder.py

加载旧 patch 中 u_
in/
u_
down/
u_
up boundary spec references。

可选复制。只用于 boundary_
wrappers.py，不再训练谱系数
decoder。

teukspec/
atlas/
patch_
expert.py

如果已有 patch expert 能稳定评估 boundary u_
spec，可复用。

可选复制；若太重，只抽取 eval
branch reference 的最小代码。

teukspec/
correctors/
film_
pirate_
corrector.py

其中的 Pirate/
FiLM block 思想可复用。

只复制 block 定义或重写为
teukfield/
models/
blocks.py。不要保留“谱解+外接
correction”的旧流程。

明确不要复制：旧 training outputs、旧 npz 结果、scripts/train_patch_spectral_decoder.py、
scripts/fit_up_param_cheb_decoder.py 作为主流程、旧 residual-correction 训练脚本、旧报告里已废弃的普通 MLP
coefficient head。旧脚本可以保留在仓库历史中，但新分支实现不要依赖它们。

Teukfield-AmpNet agent implementation plan

第3页

3. 新分支目录结构
建议创建以下目录和文件。agent 需要优先实现 skeleton，使所有 import、配置读取、forward 接口和 loss 接口跑
通；随后再逐步填充物理细节。
teukfield/
__init__.py
physics/
__init__.py
coordinates.py
# r, z, y, r_+, r_-, tortoise
factors.py
# P, h2, A_in, A_down, A_up
angular.py
# lambda loader / spheroidal eigenvalue wrapper
reduced_equation.py
# coefficients for S-equation and R-equation residual
cheb_tools.py
# quadrature/test functions/debug only
models/
__init__.py
encoders.py
# PhysicsFeatureEncoder
blocks.py
# SIREN, Fourier features, FiLM, Pirate residual blocks
local_windows.py
# FBPINN-style partition of unity windows
s_field.py
# Global S_theta neural field
amplitude_head.py
# B_inc and rho head
teukfield_ampnet.py
# integrated model
data/
samplers.py
# Sobol/LHS param sampling + radial collocation
anchors.py
# benchmark/spectral anchor data adapters
losses/
residuals.py
# strong + weak residual
boundary_consistency.py
# inner/outer spectral consistency
amplitude_losses.py
# amp anchor + phase-safe loss
total_loss.py
training/
schedule.py
# multi-stage weight schedule
trainer.py
# train loop
validation.py
# random off-grid metrics
config/
teukfield_ampnet_moderate_patch.yaml
scripts/
train_teukfield_ampnet.py
validate_teukfield_ampnet.py
docs/
agent_teukfield_ampnet_plan.pdf
agent_teukfield_ampnet_plan.md

4. 模型总图

Teukfield-AmpNet agent implementation plan

第4页

图 2：Teukfield-AmpNet 的核心是全局神经场 S_theta 和共享参数编码器。振幅 head 不独立训练，而是通过无穷远谱一致性与全局
R_theta 联合训练。

5. 数学对象和变量约定
5.1 坐标
z = r_+ / r
y = 2 z - 1
r = r_+ / z
y = 1: horizon
y = -1: infinity

训练网络可以输入 y，也可以输入 z。建议模型内部统一使用 y，因为 hard boundary wrapper 写起来更直接；物
理因子和 spectral branch wrapper 可以在 z 中计算。

5.2 参数特征
裸输入只用 a 和 omega 不够。共享物理编码器应输入包含奇异结构的特征：
p_feat = [a, log10(omega), omega, a*omega, r_plus, r_minus, kappa, Omega_H, k, lambda]
Omega_H = a/(r_+^2 + a^2)
k = omega - m*Omega_H
kappa = sqrt(1-a^2)

注意 lambda 的来源要和旧谱代码一致。若 spheroidal eigenvalue 还没有可靠 Python 实现，第一版从旧谱 solver 或
缓存数据读取。

5.3 主变量
新分支的主变量是全局 reduced field：
S_theta(y,p) = R_in(r;p) / (P(r;p) h2(r;p))
R_theta(r;p) = P(r;p) h2(r;p) S_theta(y;p)

P 和 h2 是旧分支中已经探索过的正则化因子组合。agent 不应重新发明因子，而应先复用旧分支中的定义，然后
在 docs 中记录具体 convention。

6. 模块 1：Shared Physics Encoder
PhysicsFeatureEncoder 提取参数空间共同特征，供 S-field 的 FiLM 调制和 amplitude head 共用。这样振幅和全局解
共享物理参数表示，避免两个网络学习互相不一致的参数依赖。
输入：p_feat，形状 [batch, n_feat]。
输出：latent h_p，形状 [batch, latent_dim]，建议 latent_dim=64 或 96。
结构：Fourier features for log10 omega, k, kappa + Cheb/KAN-style mixer + adaptive residual blocks。
第一版如果没有 KAN 实现，可用 Chebyshev polynomial features + gated residual block 替代。不要退化成简单浅层
MLP。
class PhysicsFeatureEncoder(nn.Module):
def forward(self, params):
raw = normalize_physics_features(params)
fourier = fourier_features(raw[:, selected_indices])
cheb = chebyshev_features(raw[:, selected_indices])
h = concat(raw, fourier, cheb)
h = adaptive_residual_blocks(h)
return h_p

7. 模块 2：Global S-field neural network
Teukfield-AmpNet agent implementation plan

第5页

坐标网络回到用户原本想要的参数 FiLM 调制坐标网络，但主干不再是普通 MLP。推荐 FBPINN-style local neural
field：把全局 y 轴分成多个重叠窗口，每个窗口一个小的 SIREN/Pirate block，最后用 partition of unity 加权求和。

7.1 表示形式
S_theta(y,p) = 1 + d_H(p)*(y-1) + (1-y)^2 * N_theta(y,p)
N_theta(y,p) = sum_j w_j(y) * N_j(xi_j(y), h_p)

这样视界端零阶和一阶行为被 hard-coded：S(1,p)=1，S_y(1,p)=d_H(p)。d_H 由旧谱/退化方程模块提供。若第
一版暂时没有解析 d_H，可先从 spectral boundary cache 读取，并在代码中明确 TODO。

7.2 Local block
LocalBlock_j:
input: xi_j(y), h_p
coordinate embedding: SIREN or Fourier features
modulation: FiLM gamma/beta from h_p
core: Pirate adaptive residual layers, alpha initialized near 0
output: complex correction [Re, Im]

7.3 推荐参数
参数

建议值

说明

J windows

6-10

moderate patch 先用 8。低频/
高频可增加。

overlap

0.35-0.50

保证窗口拼接光滑。

hidden_
dim

96

小 patch 可 64；高频可 128。

depth per block

4

每个 local block 不宜太深。

activation

sine /
gated sine

SIREN 有利于导数和局部振荡。

Pirate alpha_
init

1e-3 或 0

让网络从近似线性/
零修正状态开始。

dtype

float64 preferred

Teukolsky residual 对导数和相位敏感。

8. 模块 3：Amplitude Head
振幅 head 和 S-field 共用 PhysicsFeatureEncoder，不独立训练。它输出 B_inc 和 rho=B_ref/B_inc，而不是直接回归
B_ref。这避免低频下巨大尺度分离造成训练病态。
amp_head(h_p) -> log_abs_Binc, phase_Binc, log_abs_rho, phase_rho
B_inc = exp(log_abs_Binc) * exp(i phase_Binc)
rho = exp(log_abs_rho) * exp(i phase_rho)
B_ref = rho * B_inc

phase 可以用 cos/sin 形式输出：
phase output: (cos_phi_raw, sin_phi_raw)
normalize to unit vector
complex_phase = cos_phi + i sin_phi

推荐 amplitude head 使用低维结构化网络：Cheb/KAN-style features + residual blocks。因为振幅是纯参数函数，比
S(y,p) 更适合参数空间结构化逼近。

Teukfield-AmpNet agent implementation plan

第6页

9. 模块 4：Spectral Boundary Wrappers
旧分支的谱分支 u_in, u_down, u_up 只在边界附近使用，不再作为全局主表示。agent 应实现 boundary wrapper，
使训练时可以在指定窗口采样并计算谱一致性。

9.1 Inner consistency
R_inner_spec = A_in * u_in_spec
L_inner = mean( |P*h2*S_theta - R_inner_spec|^2 / (|R_inner_spec|^2 + eps) )
window: y in [0.5, 1] or z in [0.75, 1]

9.2 Outer consistency
R_outer_spec = B_inc_theta*A_down*u_down_spec + B_ref_theta*A_up*u_up_spec
L_outer = mean( |P*h2*S_theta - R_outer_spec|^2 / (|B_inc*A_down*u_down| + |B_ref*A_up*u_up|
+ eps)^2 )
window: z in [0.02, 0.35] for moderate patch
low frequency: use adaptive far-field window
high frequency: use two-channel/middle window extension

这一步是新方案的核心。它用可训练的振幅 head 替代旧的病态 post-hoc LS，同时仍利用谱方法在无穷远边界的
高精度行为。

10. 模块 5：Residual losses
10.1 Strong relative residual
用自动微分得到 S_y 和 S_yy，并在 reduced S-equation 或完整 R-equation 中计算相对残差。第一版建议优先在
reduced S-equation 中做 residual，因为完整 R 里 prefactor 强振荡会显著恶化 conditioning。
res = L_S[S_theta]
den = |term_2| + |term_1| + |term_0| + eps
L_res = mean( |res|^2 / den^2 )

10.2 Weak residual
强形式 residual 对二阶导点值敏感。弱形式 residual 用窗口内 test functions 积分，稳定全局结构。第一版可用
Chebyshev/Gauss-Lobatto quadrature 在每个 radial window 上计算。
For each window I_j and test function phi_q:
W_jq = integral_Ij L_S[S_theta](y,p) * phi_q(y) dy
L_weak = sum_jq |W_jq|^2 / scale_jq^2

10.3 Anchors
接受 benchmark anchor 是必要的。anchor 不应支配最终训练，但用于防止 PINN 走错物理解支。
L_R_anchor = relative error between R_theta and R_ref at sparse off-grid points
L_amp_anchor = phase-safe loss between amp_head and B_ref data if available

11. Total loss
L_total =
lambda_inner * L_inner
+ lambda_outer * L_outer
+ lambda_res
* L_res_strong
+ lambda_weak * L_res_weak
+ lambda_R
* L_R_anchor
+ lambda_amp
* L_amp_anchor
+ lambda_Abel * L_Abel_Wronskian
+ lambda_smooth* L_param_smooth
+ lambda_reg
* L_correction_regularization

Teukfield-AmpNet agent implementation plan

第7页

其中 L_outer 是联通 S-field、amplitude head、无穷远谱分支的关键项；L_res 是最终物理方程精度项；
L_R_anchor 只用于定相位、定归一化和防止错误分支。

Teukfield-AmpNet agent implementation plan

第8页

12. 训练流程图

图 3：训练流程必须采用 curriculum。不要一开始同时打开所有 loss，否则容易得到低 residual 但振幅错误或归一化错误的解。

13. 完整训练策略
Stage 0：准备 patch 与数据
先使用当前 moderate patch，而不是全域：
a_center = 0.5
a_half_width = 0.06 or 0.12
logw_center = -1.5
logw_half_width = 0.125 or 0.25
l = m = 2, s = -2

准备四类采样点：
global residual collocation: y 全域，Sobol/随机 + boundary 加密。
inner spectral consistency window: z in [0.75,1.0]。
outer spectral consistency window: z in [0.02,0.35]。
benchmark anchors: sparse pybhpt/MMA/spectral reference，必须严格 off-grid。

Stage 1：boundary + anchor warm-up
目标是让网络先学到正确物理解支。此阶段 residual 权重很小。
lambda_inner = 1.0
lambda_outer = 1.0
lambda_R_anchor = 0.5
lambda_amp_anchor = 0.2
lambda_res = 1e-3
lambda_weak = 0
steps = 5k-20k AdamW

Stage 2：逐步打开强形式 residual
lambda_res: 1e-3 -> 1.0 using cosine/linear ramp
lambda_R_anchor: 0.5 -> 0.1
lambda_inner: keep 0.5-1.0
lambda_outer: keep 1.0

Teukfield-AmpNet agent implementation plan

第9页

此阶段 S_theta 开始成为真正的 PINN 解，而不只是拟合边界和数据。每 1k steps 做一次 random off-grid residual
scan。

Stage 3：加入 weak residual 与 Abel/Wronskian 检查
weak residual 抑制二阶导点值噪声；Abel/Wronskian 用于检查 outer basis 与振幅约定是否一致。
lambda_weak = 0.05-0.2
lambda_Abel = 0.01-0.1
if L_res decreases but amplitude error increases: lower lambda_res, increase
lambda_outer/lambda_amp

Stage 4：amplitude head 精修
冻结 S-field 主干的大部分层，只训练 amplitude head、共享 encoder 最后两层、FiLM scale/bias 的最后层。
loss = L_outer + 0.2*L_amp_anchor + 0.05*L_Abel
lr = 1e-5 to 3e-5
steps = 2k-10k

Stage 5：全联合低学习率微调
解冻全部模块，低学习率联合微调。若 PyTorch L-BFGS 对复杂 batch 不稳定，先用 AdamW + cosine decay；后续
再增加 NNCG/L-BFGS 局部 refinement。

14. 配置文件模板
创建 config/teukfield_ampnet_moderate_patch.yaml：
project:
name: teukfield_ampnet
branch: research/teukfield-ampnet-20260607
physics:
s: -2
l: 2
m: 2
M: 1.0
patch:
a_center: 0.5
a_half_width: 0.06
logw_center: -1.5
logw_half_width: 0.125
model:
dtype: float64
latent_dim: 64
fourier_bands: 6
n_windows: 8
window_overlap: 0.4
local_hidden_dim: 96
local_depth: 4
activation: sine
pirate_alpha_init: 1.0e-3
amplitude_head:
type: cheb_kan_residual
hidden_dim: 64
depth: 3
sampling:
param_batch: 16
residual_y_points: 256
inner_points: 64
outer_points: 128
anchor_points: 64
loss_weights:
stage1:
inner: 1.0
outer: 1.0
R_anchor: 0.5
amp_anchor: 0.2
residual: 1.0e-3
weak: 0.0
stage2:

Teukfield-AmpNet agent implementation plan

第 10 页

residual_ramp_to: 1.0
R_anchor_ramp_to: 0.1
stage3:
weak: 0.1
Abel: 0.05
optimizer:
adamw_lr: 1.0e-4
fine_tune_lr: 3.0e-5
validation:
random_param_points: 100
random_y_points: 200
target_residual_median: 1.0e-5
target_amp_median: 1.0e-4

Teukfield-AmpNet agent implementation plan

第 11 页

15. 实现任务清单
优先级

任务

验收方式

P0

创建新分支和目录结构。

python -m compileall teukfield scripts 通
过。

P0

从旧分支复制/
重构 factors、coordinates、reduced equation 最小代码。

能在给定 a,omega,y 上计算 P,h2,A_
in,A_
down,A_
up 与 residual coefficients。

P0

实现 PhysicsFeatureEncoder、LocalWindowSet、SIREN/
Pirate/
FiLM block。

forward shape 正确，float64 可运行，
alpha 初始化接近 0。

P0

实现 SFieldNetwork，hard-code S(1)=1 和 S_
y(1)=d_
H。

数值检查边界值和导数误差 <1e-10。

P0

实现 AmplitudeHead。

能输出 B_
inc, B_
ref, rho；phase 输出归一化稳定。

P1

实现 boundary wrappers，调用旧谱参考或缓存 u_
in/
u_
down/
u_
up。

inner/
outer consistency loss 可计算。

P1

实现 strong residual loss。

对 benchmark teacher 解 residual 应很小
；对随机网络 residual 非零。

P1

实现 weak residual loss。

单元测试：quadrature shape、test
functions shape 正确。

P1

实现训练 schedule 与 trainer。

Stage 1-5 可按 config 切换，日志记录
每个 loss。

P2

实现 validation: random off-grid residual、outer amplitude consistency、
pybhpt/
MMA anchor。

输出 JSON/
CSV summary，不使用训练节点。

P2

实现 ablation: no outer loss /
no amp head /
no weak residual /
no local windows。

确认每个模块的必要性。

16. 验收指标
第一个 moderate patch 的最低验收标准：
1. Boundary hard constraints:
|S(1)-1| < 1e-10
|S_y(1)-d_H| < 1e-8
2. Strong residual:
median relative residual < 1e-5
p90 relative residual < 5e-5
max excluding endpoints < 1e-3
3. Outer consistency:
median outer relative mismatch < 1e-5
p90 < 1e-4
4. Amplitude accuracy, if anchor available:
median relerr B_inc/B_ref < 1e-4 initially
p90 < 5e-4

Teukfield-AmpNet agent implementation plan

第 12 页

5. R_in benchmark anchors:
median relative R error < 1e-5
p90 < 1e-4
6. Failure criterion:
residual low but amplitude error high -> reject, strengthen L_outer/L_amp and inspect
convention

17. 关键反模式
不要把旧 ParamChebDecoder 当主网络。它最多用于生成 boundary spec references。
不要把 multipoint LS 当最终振幅求解器。它只能做 diagnostic/pseudo-anchor。
不要一开始上全域 a in [0,1), omega in [1e-4,10]。先做 moderate patch。
不要只看训练 loss 或 structured grid 验证。必须看 random off-grid 参数点。
不要让 amplitude head 独立训练。它必须通过 outer spectral consistency 与 R_theta 联合闭环。
不要在完整 R 上优先做 residual，如果 prefactor 导致 conditioning 爆炸；先用 reduced S-equation。
不要复制旧分支的大数据输出和废弃实验脚本。新分支应清晰、轻量、可回退。

Teukfield-AmpNet agent implementation plan

第 13 页

18. 伪代码：Integrated forward
class TeukfieldAmpNet(nn.Module):
def forward(self, y, params):
# params: a, omega, lambda, etc.
feat = build_physics_features(params)
h_p = self.physics_encoder(feat)
S = self.s_field(y, h_p, feat)
B_inc, B_ref, rho = self.amplitude_head(h_p)
P = factors.P(y, params)
h2 = factors.h2(y, params)
R = P * h2 * S
return {
"S": S,
"R": R,
"B_inc": B_inc,
"B_ref": B_ref,
"rho": rho,
}

19. 伪代码：Training step
def training_step(batch, model, loss_weights):
y_res, params = batch["residual"]
out_res = model(y_res, params)
L_res = strong_residual_S(out_res["S"], y_res, params)
y_in, spec_in = batch["inner_spec"]
out_in = model(y_in, params)
L_inner = inner_consistency(out_in, spec_in, params)
y_out, spec_down, spec_up = batch["outer_spec"]
out_out = model(y_out, params)
L_outer = outer_consistency(out_out, spec_down, spec_up, params)
anchor = batch.get("anchors")
L_anchor = anchor_loss(model, anchor) if anchor is not None else 0
L_weak = weak_residual(...) if loss_weights["weak"] > 0 else 0
L_abel = abel_loss(...) if loss_weights["Abel"] > 0 else 0
L = w_inner*L_inner + w_outer*L_outer + w_res*L_res + \
w_weak*L_weak + w_R*L_anchor + w_Abel*L_abel
return L, metrics

20. Agent 的第一轮提交目标
第一轮提交不要求训练到目标精度，但必须建立正确架构。
Commit 1: branch skeleton
- new directories
- config file
- docs/agent_teukfield_ampnet_plan.md
Commit 2: physics wrappers
- coordinates, factors, reduced_equation
- tests for shapes and finite values
Commit 3: model skeleton
- physics encoder
- local windows
- S-field network
- amplitude head
- integrated model
Commit 4: losses and training script
- strong residual
- inner/outer consistency
- anchors interface
- train_teukfield_ampnet.py dry-run
Commit 5: validation script

Teukfield-AmpNet agent implementation plan

第 14 页

- random off-grid sampling
- JSON metrics
- smoke benchmark using synthetic / cached anchors

21. 参考思想，不作为硬依赖
本方案借鉴以下方向：DeepONet/PINO 的参数到解函数 operator learning；FBPINN/XPINN 的局部分区；
VPINN/hp-VPINN 的弱形式 residual；SIREN/Fourier features 的高频和导数表达能力；PirateNets 的 adaptive
residual initialization；KAN/Cheb-KAN 在低维参数函数上的结构化逼近；MST/Teukolsky 系列方法在边界渐近和
benchmark 上的高精度优势。具体代码实现不要求引入所有外部库，优先实现可控、轻量、可测试的自定义模
块。
理论背景上，黑洞微扰方法和 Teukolsky 方程是 EMRI/IMRI 高精度波形建模的重要基础；Sasaki-Tagoshi 综述强
调 Teukolsky 与 MST/解析展开在黑洞微扰中的核心作用。上传文献中也多次强调，Teukolsky-based 波形计算精
确但昂贵，因此快速且保真的 surrogate/近似模型具有实际意义。

Teukfield-AmpNet agent implementation plan

第 15 页

22. 最终交付物
agent 完成后，新分支至少应包含：
docs/agent_teukfield_ampnet_plan.md
config/teukfield_ampnet_moderate_patch.yaml
teukfield/physics/*.py
teukfield/models/*.py
teukfield/losses/*.py
teukfield/training/*.py
scripts/train_teukfield_ampnet.py
scripts/validate_teukfield_ampnet.py
tests/test_teukfield_shapes.py
tests/test_teukfield_boundary.py
tests/test_teukfield_losses.py

README 或 docs 中必须写清楚：
当前实现是 global S-field + amplitude head，不是 spectral coefficient decoder。
旧谱模块只用于 boundary reference/anchor/diagnostics。
最终验收看 random off-grid residual、R benchmark、amplitude consistency，而不是训练 loss。

本 PDF 到此结束。agent 应从第 2 节开始执行，不要跳过分支隔离和最小复制规则。

Teukfield-AmpNet agent implementation plan

第 16 页

