# comfyui-b70-xpu

**让最新版 ComfyUI 在 Intel Arc Pro B70 上跑起来，并且真的走在快路径上。**

[English](README.md)

Intel 官方有一个给 Arc 显卡用的 ComfyUI 镜像（`intel/llm-scaler-omni`），速度不错，但已发布的镜像
（0.2.0-b1）停在 ComfyUI 0.31 和 torch 2.11，新模型需要更新的 ComfyUI。这个仓库是我们在一张
Arc Pro B70（32 GB）上实际在用的方案：ComfyUI 上游最新版 + torch 2.13，把 Intel 的 XPU 加速层搬过来，
修好升级时坏掉的地方，再加上我们自己写的算子和修复。

仓库的目录结构照着 [intel/llm-scaler](https://github.com/intel/llm-scaler) 的 `omni/` 目录来，
镜像**从 Intel 已发布的镜像构建**，所以部署只要一条 `bash build.sh`，运行命令也和官方文档里的一样。

> Intel 的 `main` 分支上现在有一个尚未发布的 `0.2.0-b2` 源码构建，对应 ComfyUI 0.39 和 torch 2.14。
> 我们没有构建和测过它。这里所有对比的对象都是已发布的 `0.2.0-b1` 镜像；等 `0.2.0-b2` 正式发布，
> 这个对比应该重做。

支持并实测了四个模型：**MiniMax H3**（带声音的视频）、**Krea-2 Turbo**、
**Qwen-Image 2.1**（含 Viggle Turbo）、**MiniMax Music 3**。

## 结果速览

一张 Arc Pro B70，功耗上限 230 W。单位是每个任务的秒数，取服务端记录的执行时间。
「换提示词」才是一个真实任务的耗时，原因见[数字是怎么测的](#数字是怎么测的)。

**和 Intel 官方镜像比，同一张卡、同一天、同一份工作流（H3 图生视频，6 步，124 帧）：**

| | 官方镜像 0.2.0-b1 | 本仓库 | |
|---|---|---|---|
| 480×864，只换种子 | 59.2 – 59.4 s | 55.7 – 56.0 s | −6% |
| 480×864，**换提示词** | 87.5 – 87.8 s | 69.9 – 72.4 s | **−17% 到 −20%** |
| 704×1280，只换种子 | 181.5 – 181.7 s | 169.5 – 169.8 s | −7% |
| 704×1280，**换提示词** | 210.9 – 213.3 s | 185.1 – 185.2 s | **−12%** |

**另外三个模型**，比的是本方案上浮点权重和官方 INT8 ConvRot 权重。
（官方镜像上我们只测了 H3。Krea-2 在官方镜像上也能跑，但我们没有在那上面测过；Qwen-Image 2.1 在它的 ComfyUI 0.31 上加载不了。）

| | 浮点权重 | INT8 ConvRot 权重 | |
|---|---|---|---|
| Krea-2 Turbo，8 步，1024²，只换种子 | 9.9 s | 6.9 – 7.3 s | −28% |
| Krea-2 Turbo，换提示词 | 14 – 16 s | 7.6 s | −50% |
| Qwen-Image 2.1，25 步，1280×720 | 15.4 – 15.8 s | 10.3 – 10.6 s | −33% |
| Qwen-Image 2.1，25 步，2048² | 89.7 – 89.8 s | 67 – 68 s | −25% |
| Qwen-Image 2.1 + Viggle Turbo，6 步，1280×720 | 5.1 s | （不建议） | |
| Music 3，两分钟，30 步，只换种子 | 79.0 s | 53.2 s | −33% |
| Music 3，新歌 | 110 – 113 s | 87 – 90 s | −21% |

完整表格（含升级前的环境和每一条注意事项）见[测试数据](#测试数据)。

## 为什么做这件事

1. **要跑新模型。** Qwen-Image 2.1 需要 ComfyUI 0.37 以上。官方已发布的镜像是 0.31.0，加速层钉死在
   torch 2.11（`omni_xpu_kernel` 拒绝别的 torch 小版本，comfy-kitchen 的 XPU 分支基于 0.2.28，
   节点里有五处检查 `torch == 2.11`）。换到 ComfyUI 最新版，三层都得重做。
2. **不能丢速度。** 不带这层加速的 ComfyUI + torch XPU，H3 大约慢 1.6 倍。而且升级之后，
   加速层可以「看起来正常」却被绕过：我们从 0.34 升到 0.39 时出现了三处静默回退，启动日志全绿。
3. **这张卡还有余量的地方继续挖。** ComfyUI 在 Intel 显卡上从不启用锁页内存；几个算子还有空间；
   四个模型里有三个因为和模型本身无关的原因掉出了快路径。

## 快速开始（Docker）

要求：Linux 主机，Arc B 系列显卡且驱动正常，装有 Docker。自己构建的话还需要 BuildKit、
约 45 GB 空闲磁盘（官方基础镜像 21 GB）以及构建时能联网。

直接拉取已发布的镜像（10.8 GB，17 层，下载量约 5.5 GB）：

```bash
docker pull gerrywang/comfyui-b70-xpu:0.1.0-b1
```

或者自己构建：

```bash
git clone https://github.com/igerry/comfyui-b70-xpu.git
cd comfyui-b70-xpu/omni
bash build.sh                 # 得到 comfyui-b70-xpu:0.1.0-b1-comfyui-bmg
```

构建过程：拉取 `intel/llm-scaler-omni:0.2.0-b1`，加装 Python 3.13 和 `torch 2.13.0+xpu`，
在一个用完即弃的阶段里装 2026.0 版 DPC++ 编译器，用本仓库的源码编译 `omni_xpu_kernel`
（16 线程约 3 分钟），按固定提交检出 ComfyUI，装上 Kitchen 的 XPU 后端和 `ComfyUI-OmniXPU` 节点。
最后删掉官方镜像原有的 ComfyUI 0.31 / torch 2.11 / oneAPI 那套环境，把结果写进一个全新的镜像；
从官方镜像保留下来的是操作系统和显卡的用户态驱动。Python 环境被拆成 16 层，这样上传和下载都能并行。

先验证，再运行。挂载点和参数与官方镜像相同：

```bash
IMAGE=gerrywang/comfyui-b70-xpu:0.1.0-b1      # 或者你本地构建的 comfyui-b70-xpu:0.1.0-b1-comfyui-bmg

docker run --rm --device=/dev/dri "$IMAGE" /llm/entrypoints/validate_comfyui_image.sh

docker run -itd --name comfyui \
    --device=/dev/dri --network=host --shm-size=16g --memory=96g \
    -v /path/to/comfyui_models:/models/host:ro \
    -v /path/to/comfyui_input:/data/input \
    -v /path/to/comfyui_output:/data/output \
    -v /path/to/comfyui_user:/data/user \
    "$IMAGE"
```

`--memory` 填 ComfyUI 可以指望的内存：物理机的内存；Docker 跑在 LXC 里时填 LXC 的内存大小
（测试机是 96 GB）。

之后访问 `http://<主机>:8188`。默认命令是 `/llm/entrypoints/start_comfyui.sh`
（直接启动；跟在它后面的参数会传给 `main.py`）。也可以像用官方镜像那样自己写 `python main.py ...`。

- 我们实际运行时多加了 `--privileged`（测试机是 LXC 容器，必须加；官方早先的说明也是这么写的）。
  不加 `--privileged` 的情况没有测过，尤其是锁页内存。
- **在 LXC 里运行时，要把 LXC 的内存大小传给容器：`--memory=96g`**（换成你自己 LXC 的值）。
  嵌套在 LXC 里的 Docker 看到的是物理机的内存，不是 LXC 的限额：在我们的机器上，锁页内存预算
  被算成了 917 GB，而不是 86 GB。ComfyUI 认 `--memory` 设下的 cgroup 限额，预算会变成这个值的 90%。
  直接跑在物理机或普通虚拟机上不需要这一条。
- **主机内存。** 锁页内存会给每个加载过的模型保留一份锁页副本，上限是内存的 90%。
  只跑 H3 就要约 52 GB，测试机是 96 GB。内存不够就加 `-e OMNIXPU_PIN_MEMORY=0`。
- **自定义节点。** 把单个节点挂到 `/llm/ComfyUI/custom_nodes/<名字>`；不要整个目录覆盖挂载，
  里面有 `ComfyUI-OmniXPU`。节点的 Python 依赖装进镜像的环境：`docker exec comfyui pip install ...`。

更多：[omni/README.md](omni/README.md)、[构建细节与裸机安装](omni/docs/IMAGE_BUILD.md)、
[模型与开关](omni/docs/COMFYUI.md)（均为英文）。

### 确认它真的生效

启动日志全绿不算数（见[兼容性修复](#1-comfyui-039-的兼容性修复)）。
`validate_comfyui_image.sh` 检查静态的部分。服务跑起来之后看这三样：

```bash
docker logs comfyui 2>&1 | grep -E "OmniXPU|Native ops"
```

1. 有这一行：`[OmniXPU] omni_xpu_kernel 0.2.0b1+torch213.bmg - available: sdp, norm, rotary, linear_fp8, int8`
2. `Native ops: ...` 这一行里有 `int8_tensorwise` 和 `convrot_w4a4`。如果它们出现在
   `emulated ops` 里，说明 INT8 权重正在被反量化成浮点来算。
3. 跑过一个工作流之后，`python3 omni/benchmarks/xpu_status.py` 会打印路由计数；注意力调用应该走
   `cute` / `esimd`，而不是回退。

### 模型文件

模型不会自动下载。下面是基准脚本用到的文件名，都是给 ComfyUI 重新打包的官方文件。

| 模型 | `diffusion_models/` | `text_encoders/` | `vae/` | `loras/` |
|---|---|---|---|---|
| MiniMax H3 | `minimax_h3_fl2va_pruned_int8_convrot` | `MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot` | `minimax_h3_video_vae_fp16`（或 `_int8_convrot`）、`minimax_h3_audio_vae_fp32` | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui`（社区 Turbo LoRA） |
| Krea-2 Turbo | `krea2_turbo_bf16` 或 `krea2_turbo_int8_convrot` | `qwen3vl_4b_bf16` | `qwen_image_vae` | |
| Qwen-Image 2.1 | `qwen_image_2.1_bf16` 或 `_int8_convrot` | `qwen3vl_8b_bf16` 或 `_int8_convrot` | `qwen_image_2.1_vae_bf16` | `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256`，外加 Viggle 的 `viggle_turbo.py` 放进 `custom_nodes/` |
| Music 3 | `minimax_music3_dit_fp16` 或 `_int8_convrot` | `minimax_music3_text_encoder_pruned_bf16` | `minimax_music3_dav` | |

这张卡上各模型该选哪种权重：

| 模型 | 选 | 原因 |
|---|---|---|
| H3 | 主模型 INT8（32 GB 里唯一实用的）。视频 VAE：480p 用 fp16，704×1280 用 INT8 | 见下面 H3 的表 |
| Krea-2 Turbo | **INT8** | 快 28%；构图几乎重合；主模型和文本编码器能同时留在显存里，换提示词不用再换进换出 |
| Qwen-Image 2.1 原版 25 步 | **INT8** | 快 33%；画质同档（和浮点比 PSNR 35 dB） |
| Qwen-Image 2.1 + Viggle Turbo | **bf16** | INT8 上人脸发灰发糊 |
| Music 3 | **DiT 用 INT8，文本编码器用 bf16** | DiT：快 33%，音频相关系数 1.000。文本编码器换 INT8 不更快，而且音频会变（相关系数 0.854） |

## 我们改了什么

涉及三层上游项目，每一行都写明改动在哪。

| 层 | 上游 | 在本仓库里 |
|---|---|---|
| 原生算子（SYCL / ESIMD / CUTLASS-SYCL） | `omni_xpu_kernel`、`sycl-tla` | `omni/omni_xpu_kernel/`（完整源码）、`omni/patches/sycl-tla-fmha-b70.patch` |
| 算子分发 | `comfy-kitchen` + Intel 的 XPU 后端分支 | `omni/patches/comfy_kitchen-0.2.37-xpu.patch` |
| 接入 ComfyUI | `ComfyUI-OmniXPU` 节点 | `omni/ComfyUI-OmniXPU/`（完整源码）；我们写的模块在它的 `fixes/` 里 |

`omni/patches/vs-intel-0.2.0-b1/` 里是这两个源码目录相对官方 `0.2.0-b1` 镜像内源码的完整 diff，方便审阅。

### 1. ComfyUI 0.39 的兼容性修复

这三处出问题时，启动日志都显示每个挂钩已生效。

| 上游改了什么 | 在 XPU 上的后果 | 修复 |
|---|---|---|
| 新增的 `model_management.supports_int8_compute()` 对 Intel XPU 返回 `False` | 每个 INT8 线性层被标成「全精度」，权重每次调用都反量化成 bf16。H3 采样从每步 6.2 秒变成 7.3 秒，不报错 | `ComfyUI-OmniXPU/fixes/xpu_int8_compute.py` |
| comfy-kitchen 的 `int8_linear` 多了四个参数，分发器总是全传 | INT8 层第一次运行时 XPU 后端报 `TypeError` | XPU 后端的包装函数接收并处理这些参数 |
| `pin_memory()` 多了 `evict_active` 参数 | Qwen-Image 2.1 第一步就崩（只有它的 KV 缓存会调到这里，H3 和 Krea-2 碰不到） | `ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py` |

移植还包括：给 torch 2.13 / Python 3.13 编译 `omni_xpu_kernel`（编译器和 oneDNN 换到 2026.0 这一代、
并行构建、放开 torch 小版本白名单），把 Intel 的 XPU 后端重新接到 comfy-kitchen 0.2.37 上，
放宽节点里的 torch 版本检查。

### 2. 锁页内存（任何 Intel 显卡、任何模型都适用）

ComfyUI 只在 `is_nvidia() or is_amd()` 时设置锁页内存，并且用 CUDA 的接口来锁页。
在 XPU 上，每次装载和卸载模型走的都是可分页拷贝。本机实测：主机到显卡，可分页 1.87 GB/s，
锁页 26.75 GB/s；显卡到主机，2.14 对 9.56 GB/s。

`ComfyUI-OmniXPU/fixes/xpu_pinned_memory.py` 按 ComfyUI 自己的公式设预算，用「拷贝再换存储」的方式锁页
（torch-xpu 没有注册接口），并让整模型卸载直接落进锁页内存。

- H3 每次换提示词（文本编码器装入、卸出，主模型再装回）：**81 – 83 秒 → 69 – 74 秒**。
- 只换种子的重跑不变（没有东西要换）。
- 在 96 GB 的主机上四个模型轮换时有得有失：有的切换快 22 – 24 秒，有一处慢 14 – 21 秒。
  我们的判断是主机内存吃紧（锁页副本加页缓存超过内存，某个模型只能从磁盘重读），
  这是从两组序列的对比推出来的，没有直接采样到内存占用。

官方镜像的两种启动方式都没有锁页内存。

### 3. 非二维输入也走 INT8 快路径（Krea-2、Qwen-Image 2.1、Music 3）

内核里「旋转 + 量化 + 矩阵乘」的融合路径只接受二维输入。H3 传的是 `(M, K)`；其他模型传的是
`(B, N, K)`，于是悄悄走了通用路径（先生成 bf16，再旋转、量化、矩阵乘）。
现在 XPU 后端会先把连续的输入摊平成 `(行, K)`。

| 整跑，INT8 权重 | 改前 | 改后 |
|---|---|---|
| Qwen-Image 2.1，25 步，1280×720 | 11.6 s | 10.3 s |
| Krea-2 Turbo，1024² | 7.4 s | 6.9 s |
| Music 3 | 66.8 s | 61.8 s |

Qwen-Image 2.1 同种子重跑仍然逐像素相同。

### 4. Music 3

| 改动 | 开关 | 整跑 | 输出 |
|---|---|---|---|
| 把每一步里等长的窗口并成一批（两分钟的歌每步有 30 个重叠窗口，上游是一个一个算的） | `OMNIXPU_MUSIC3_BATCH` | 87.7 → 83.5 s | 相关系数 1.0000 |
| 部分 RoPE 改在稠密拷贝上做。XPU 上对跨步张量做逐元素运算约 80 GB/s，稠密张量约 450 GB/s | `OMNIXPU_MUSIC3_ROPE` | fp16 83.5 → 79.0 s；INT8 61.8 → 58.2 s | 逐样本相同 |
| GLU 门控并进 INT8 下投影：SiLU × 乘 + 旋转 + 量化一遍完成，直接输出 int8（需要内核的第 9 阶段补丁） | `OMNIXPU_MUSIC3_GLU` | INT8 58.2 → 53.2 s | 相关系数 0.99998 |

合计：升级前 88.4 秒 → 79.0 秒（fp16）→ 53.2 秒（INT8 DiT）。

### 5. Qwen-Image 2.1：大尺寸注意力走 CUTE 算子

Qwen-Image 2.1 的注意力里键比查询多（前缀 + 图像行），适配层原先不会把它交给 CUTLASS 算子。
按单次调用实测，这个算子在小形状上输、大形状上赢（查询 3600 / 键 3650：0.93 倍；
加 3 张参考图：1.09 倍；加 10 张：1.36 倍；2048² 文生图：1.43 倍），所以只在
`kv_len ≥ 7168` 时启用（`OMNIXPU_QWEN21_CUTE_MIN_KV`）。2048²、INT8、25 步：**76.7 → 67.8 秒**；
1280×720 不变且逐像素相同。多参考图的整跑没有测，只有单次调用的数字。

### 6. H3 的原生算子（9 月的成果，沿用并重新验证）

以下是按真实形状测的单层微基准；单块微基准加起来 140.3 毫秒，和实际剖面的每层 141.6 毫秒对得上。

| 算子 | 改了什么 | 微基准 | 数值 |
|---|---|---|---|
| 融合 ConvRot + INT8 量化 v2 | Intel 原核慢在第一级 Hadamard 用了 64 次标量蝶形。改成二维视图上的 4 个向量操作；16 个线程分一行；旋转后的值留在寄存器里给第二遍用 | K=14336：4.13 → 1.43 ms（490 GB/s） | 缩放系数一致；每一亿个 int8 里有 1–3 个差 1 个最低位 |
| 注意力：惰性重缩放 | 子组内没有哪一行的最大值变化时，跳过输出累加器上的 128 次乘法 | 71.7 → 70.4 ms | 逐位相同 |
| RMSNorm 并入 AdaLN 调制 | 缩放和偏移在归一化那一遍里完成 | 2.03 → 0.67 ms | 1 个最低位 |
| RoPE 子组算子 | 每行一个 16 路子组，不用共享本地内存，不用栅栏 | 2.75 → 1.90 ms（492 GB/s） | 最大 1 个最低位 |
| 「生产者」融合：SiLU×乘 → int8，以及（残差 +）归一化 + 调制 → int8 | 每个 INT8 矩阵乘前面的那个算子直接写出 int8，不再先写 bf16 再读回来量化 | 4.03 → 2.20 ms；2.23 → 1.18 ms | 逐位相同 / 每亿约 1500 个差 1 个最低位 |
| H3 视频 VAE 注意力分块 | 只给这条路由用 KV 块 64 + 3 级流水 | 0.346 → 0.292 ms | 40.9 dB |
| 适配层：VAE q/k 视图上的归一化 | 每条视频 7560 次调用不再回退到 torch | 0.33 → 0.21 ms | |

480p 整跑、只换种子的耗时由此从 58.0 降到 56.3 – 56.6 秒。其中两项（整块融合和 AdaLN 融合）
在本仓库里**默认关闭**，原因见下一节。

### 7. 可复现性：同一个种子出同一条视频

全部打开时，H3 同一个种子每次跑出来的视频肉眼可见地不同（两次之间 30.8 – 37.7 dB）。
我们把每种配置各跑六次，逐位比较潜变量：

| 配置 | 出现几种潜变量 | 潜变量相对差 | 视频 |
|---|---|---|---|
| 纯上游（我们的代码和 XPU 后端全关） | 4 次出 4 种 | 5e-6 或 5e-2 | 2 次不同（36.4 dB） |
| 开 XPU 后端，关两个 H3 融合 | 6 次出 3 种 | 约 1e-6 | **6 次全部相同** |
| 全开 | 6 次出 4 种 | 4e-2 … 1.2e-1 | 4 次不同 |

源头在上游：每次重新装载模型之后会出现约 1e-6 的扰动，并且落在几种固定结果之一上。
把我们的代码全关掉它也在。我们那两个 H3 层内融合把它放大到 4 – 12%；关掉融合后它停在 1e-6，
画面逐帧相同。解码是确定的，分叉发生在采样。

所以 `OMNIXPU_BLOCK_FUSED` 和 `OMNIXPU_ADALN_FUSED` 默认是 `0`。代价是每条视频不到 1 秒
（55.7 – 56.0 秒对 54.9 – 55.2 秒）。Qwen-Image 2.1 无论开关都逐像素相同。
Krea-2 不相同，开不开我们的代码都一样。上游那个扰动的来源还没查清。

## 测试数据

### 测试机

| | |
|---|---|
| 显卡 | Intel Arc Pro B70，32 GB，功耗上限 230 W（出厂值） |
| 主机 | AMD EPYC 7C13，96 GB 内存，Proxmox 上的非特权 LXC 容器，Debian 13（内核 7.0.14） |
| 驱动 | compute-runtime 26.22.38646.4，Level Zero 1.31.0 |
| 软件 | ComfyUI 上游 `a4b5a045`（0.39.0），torch 2.13.0+xpu，Python 3.13，comfy-kitchen 0.2.37 + XPU 后端，`omni_xpu_kernel 0.2.0b1+torch213.bmg`（由本仓库构建） |
| 日期 | 2026-10-09 和 10-10。官方镜像是同一天重跑的 |

### 数字是怎么测的

- **ComfyUI 按输入缓存节点的输出。** 只换种子重跑会跳过文本编码器以及围绕它的模型换进换出。
  这就是「只换种子」（常驻）的数字，它主要反映算子的速度。真实任务每条提示词都不同：
  文本编码器装入、运行、卸出，主模型再装回。H3 上两者相差 14 – 30 秒。**请比「换提示词」的数字。**
- 时间取服务端 `/history` 里的 `execution_start` 到 `execution_success`，不含排队和轮询。
- H3 的跑间波动约 ±1 秒。范围是 2 – 3 次的最小值到最大值。标了「单次」的就是只跑了一次。
- 「首跑」指显卡上之前是别的模型时的第一次（含从页缓存装载）。
- 脚本：`omni/benchmarks/suite.sh` 对一个运行中的服务跑完上面两张速览表里的所有项目。

### MiniMax H3（INT8 主模型 + Turbo LoRA，6 步，124 帧带声音）

| | 官方镜像 | 升级前（ComfyUI 0.34） | **本仓库，fp16 视频 VAE** | 本仓库，INT8 视频 VAE |
|---|---|---|---|---|
| 480×864，只换种子 | 59.2 – 59.4 | 55.9 – 56.1 | **55.7 – 56.0** | 52.3 – 52.4 |
| 480×864，换提示词 | 87.5 – 87.8 | 69.7 – 74.7 | **69.9 – 72.4** | 77.3 – 78.7 |
| 704×1280，只换种子 | 181.5 – 181.7 | 169.9 – 170.0 | **169.5 – 169.8** | 163.0 – 163.2 |
| 704×1280，换提示词 | 210.9 – 213.3 | 185.4 – 185.5 | **185.1 – 185.2** | 176.8 – 177.2 |

- 「升级前」已经包含 9 月做的算子和锁页内存，所以和现在这一列持平：对 H3 来说，
  10 月的工作是在 ComfyUI 升级之后保住了速度，没有再提速。
- INT8 视频 VAE 在只换种子时总是更快，704×1280 换提示词时也更快。但 480×864 换提示词时没有收益：
  裸机 77 – 79 秒（两次，比 fp16 的 70 – 72 秒慢），容器内 73.9 秒（一次）。只换种子时的收益为什么
  带不过来，原因还没找到；480p 生产继续用 fp16 的 VAE。两种 VAE 同种子的 PSNR 是 37.8 dB。

H3 的其他工作流（秒；没写范围的都是单次）：

| | 升级前 | 本仓库，fp16 VAE | 本仓库，INT8 VAE |
|---|---|---|---|
| 第三方 Turbo 采样器节点，刚换完模型 / 再跑一次 | 69.4 / 57.0 | 73.7 / 61.4 | 93.3 / 57.7 |
| MultiRef 双镜，2 × 362 帧，euler 链 | 562.3 | 556.2（更早一次是 510） | 496.0 |

### Krea-2 Turbo（8 步）

| | 升级前，bf16 | 本仓库，bf16 | 本仓库，INT8 |
|---|---|---|---|
| 1024²，只换种子 | 9.9 – 10.2 | 9.9 | **6.9 – 7.3** |
| 1024²，换提示词 | 13.3 – 13.4 | 14 – 16 | **7.6** |
| 1536×1024 | 19.8 | 23.2 | **11.8** |

bf16 权重下，26 GB 的主模型和文本编码器放不到一起，每次换提示词都要换进换出。
INT8 的主模型是 13.5 GB，两者可以同时驻留。

### Qwen-Image 2.1

ComfyUI 0.34 和官方镜像都加载不了这个模型，所以没有「之前」这一列。

| | bf16 | INT8 |
|---|---|---|
| 原版，25 步，1280×720 | 15.4 – 15.8 | **10.3 – 10.6** |
| 原版，25 步，2048×2048 | 89.7 – 89.8（首跑 126.6） | **67 – 68** |
| Viggle Turbo，6 步，1280×720 | **5.1** | 4.2 – 4.4，人脸变差 |
| Viggle Turbo，6 步，2048×2048 | **27.5 – 29.4** | 没测 |

### MiniMax Music 3（两分钟，30 步）

| | 升级前，fp16 | 本仓库，fp16 DiT | 本仓库，INT8 DiT |
|---|---|---|---|
| 同歌词，换种子 | 87.9 – 88.0 | 79.0 | **53.2** |
| 新歌（换歌词 / 换文本编码器种子） | 113 – 121 | 110 – 113 | **87 – 90** |

一首新歌的构成：自回归生成词元约 26 秒，DiT 采样（fp16 80 秒 / INT8 57 秒），
解码和保存约 6 秒。每个进程的第一首更慢（fp16 上测到过一次 186.6 秒）。

### Docker 镜像的验证

上面的数字来自裸机安装。之后我们用本仓库构建出镜像（`comfyui-b70-xpu:0.1.0-b1-comfyui-bmg`，
**10.8 GB**），在同一台机器上跑了 `omni/benchmarks/suite.sh`，启动参数是 `--privileged --device=/dev/dri
--network=host --shm-size=16g --memory=96g`。`validate_comfyui_image.sh` 全部通过；整个过程容器日志里的错误数：0。

| 单位：秒 | 裸机 | 容器内 |
|---|---|---|
| H3 480×864，只换种子 | 55.7 – 56.0 | 55.1 – 55.7 |
| H3 480×864，换提示词 | 69.9 – 72.4 | 73.8 |
| H3 480×864，INT8 视频 VAE，只换种子 / 换提示词 | 52.3 – 52.4 / 77.3 – 78.7 | 52.8 / 73.9 |
| H3 704×1280，只换种子 / 换提示词 | 169.5 – 169.8 / 185.1 – 185.2 | 171.4 / 187.4 |
| Krea-2 Turbo bf16，只换种子 / 换提示词 | 9.9 / 14 – 16 | 9.9 – 10.0 / 12.7 |
| Krea-2 Turbo INT8，只换种子 | 6.9 – 7.3 | 6.9 |
| Qwen-Image 2.1 bf16，25 步，1280×720 | 15.4 – 15.8 | 15.8 |
| Qwen-Image 2.1 + Viggle Turbo，6 步 | 5.1 | 5.1 |
| Qwen-Image 2.1 INT8，1280×720 / 2048² | 10.3 – 10.6 / 67 – 68 | 10.3 / 67.5 – 69.8 |
| Music 3 fp16 DiT，同歌词 | 79.0 | 78.0 |
| Music 3 INT8 DiT，同歌词 / 新歌 | 53.2 / 87 – 90 | 53.1 – 55.0 / 75.2 – 86.3 |

容器内每项只跑了一到两次。在跑间波动范围内，镜像复现了裸机安装的结果。加了 `--memory=96g` 后，
容器内的锁页内存预算是 80 GB；不加的话，在这台 LXC 主机上会被算成 917 GB。启动日志的 `Native ops`
里有 `convrot_w4a4` 和 `int8_tensorwise`。镜像里的显卡驱动是官方镜像自带的，比主机上的旧
（显卡名显示为 `Intel(R) Graphics [0xe223]`），没有看到速度差异。发布到 Docker Hub 的镜像是之后重建的，
内容相同，只是拆成了 17 层；推送前它通过了验收，并且每个模型各跑通了一个任务（H3 只换种子 55.8 秒）。
容器内没跑的：Turbo 采样器节点和
MultiRef 两个工作流（第三方节点），以及不带 `--privileged` 的情况。

## 开关

环境变量，启动时读取，`0` 为关。

| 变量 | 默认 | 作用 |
|---|---|---|
| `OMNIXPU_PIN_MEMORY` | 1 | 锁页内存 |
| `OMNIXPU_INT8_COMPUTE` | 1 | 告诉 ComfyUI：XPU 原生计算 INT8 |
| `OMNIXPU_BLOCK_FUSED` | **0** | H3：整个 DiT 块的生产者融合 |
| `OMNIXPU_ADALN_FUSED` | **0** | H3：RMSNorm + AdaLN 融合 |
| `OMNIXPU_MUSIC3_BATCH` | 1 | Music 3：等长窗口并批 |
| `OMNIXPU_MUSIC3_ROPE` | 1 | Music 3：稠密的部分 RoPE |
| `OMNIXPU_MUSIC3_GLU` | 1 | Music 3：GLU 并进 INT8 下投影 |
| `OMNIXPU_QWEN21_CUTE` / `OMNIXPU_QWEN21_CUTE_MIN_KV` | 1 / 7168 | Qwen-Image 2.1：键长超过该值时注意力走 CUTE |
| `OMNIXPU_CONVROT_FUSED` / `OMNIXPU_CONVROT_SPLIT` | 1 / 16 | 融合旋转量化算子 / 每行的线程数 |
| `OMNI_H3_RMS_ROPE_SG` | 1 | RoPE 子组算子（0 = Intel 原版） |
| `OMNIXPU_DEBUG` | 0 | 打印每一次算子路由和回退 |

改动任何一条数值路径，同一个种子的输出都会变（6 步 Turbo 采样对 1 个最低位的差异是混沌的）。
要复现旧的输出，把开关恢复成当时的样子。

## 没有用的方向

记下来，免得别人再走一遍。

- **torch 2.14.1。** 搭了完整的并行环境（需要 C++20、编译器 2026.1、oneDNN 2026.0.2）。
  四个模型在稳定状态下的耗时和 2.13 相同，没有模型依赖它。2.14 新增的 XPU 图捕获，
  对由几百个小算子组成的负载有 4 – 13 倍，对算力受限的负载是 1.0 倍；这四个模型里没有哪个阶段用得上。
- **FP8 权重。** 这张卡没有 FP8 矩阵乘。原生 FP8 路径的速度就是 bf16 的速度
  （130 对 129 TFLOPS），INT8 是 207 – 238。在这张卡上要快就用 INT8。
- **更快的注意力算子。** H3 的注意力算子已经是同一块芯片上持续 bf16 矩阵乘的 82%。
  分块大小、流水级数、把 Q 留在寄存器里、拆依赖链、换子组布局：全部持平或更慢。
  这张卡顶在 230 W 的上限上，所以时间 = 周期数 ÷ 时钟，而时钟取决于数据里有多少比特在翻转
  （随机数据 70.8 毫秒 / 1983 MHz；全零数据 52.5 毫秒 / 2633 MHz，指令数相同）。
  用常数或低熵数据测注意力算子，结果会虚高约 11%。上游 `sycl-tla` 主线在这张卡上比钉住的版本慢
  （它自带的示例 147 对 63 毫秒）。
- **q/k/v 共用一次激活量化。** 微基准里快 13%，但它要靠张量的版本号，而 ComfyUI 在
  `torch.inference_mode()` 下运行，那里没有版本号。已撤回。凡是碰张量元数据的改动，
  微基准都要在 `inference_mode` 里跑。
- **Music 3 的文本编码器换 INT8**、**Music 3 的注意力走 CUTE**（序列太短）、
  **SwiGLU 并进矩阵乘的后处理**（更慢）、**官方的 `--enable-dynamic-vram` 启动方式**
  （XPU 上最慢的启动方式：直接启动 59 秒的任务它要 90.6 秒）。

## 适用范围和局限

- 只在**一张** Arc Pro B70 上测过。算子是按 `bmg` 编译的，B580 和 B60 是同一个目标，
  但这里没有在它们上面跑过。锁页内存和兼容性修复与显卡型号无关。
- 只测了上面四个模型。别的模型也会经过同样的几层，但没有测过。
- 整套版本是钉住的（ComfyUI 提交、comfy-kitchen 0.2.37、torch 2.13.0）。换更新的 ComfyUI
  要做 [omni/docs/IMAGE_BUILD.md](omni/docs/IMAGE_BUILD.md#upgrading-comfyui-later) 里的检查。
- 功耗上限保持在额定的 230 W，没有动。
- 这是一套自己在用的方案，按现状公开，不是持续维护的产品。

## 目录

和 `intel/llm-scaler` 同样的形状：

```
README.md、README.zh-CN.md   本页
Releases.md                  发布列表
omni/
  build.sh                   构建镜像（版本号读 omni/VERSION）
  docker/Dockerfile          FROM intel/llm-scaler-omni:0.2.0-b1
  entrypoints/               start_comfyui.sh、validate_comfyui_image.sh
  configs/                   comfyui_host_models.yaml（模型挂在 /models/host）
  tools/                     validate_comfyui_image.py
  omni_xpu_kernel/           原生算子，含我们改动的完整源码
  ComfyUI-OmniXPU/           ComfyUI 节点，含我们的适配和 fixes/ 的完整源码
  patches/                   comfy-kitchen XPU 后端、sycl-tla；vs-intel-0.2.0-b1/ 是我们的改动量
  benchmarks/                表格背后的脚本；kernel/ 是微基准和探针
  docs/                      IMAGE_BUILD.md、COMFYUI.md、history/（逐步的补丁脚本）
```

## 发布

发布方式照官方的惯例：一个版本号、一个 git 标签 `omni-<版本号>`、一个 Docker 镜像
（Docker Hub 上的 `gerrywang/comfyui-b70-xpu:<版本号>`）。
版本号是我们自己的，和官方的编号无关；每个版本会写明它建在哪个官方镜像之上
（`0.1.0-b1` 建在 `intel/llm-scaler-omni:0.2.0-b1` 之上）。见 [Releases.md](Releases.md)。

## 许可和致谢

Apache-2.0，与 `intel/llm-scaler` 相同（`LICENSE`）。我们对上游源码改了什么、以及其中 BSD-3-Clause
的部分，列在 [NOTICE.md](NOTICE.md)。

这项工作建立在这些项目之上：[intel/llm-scaler](https://github.com/intel/llm-scaler)
（`omni_xpu_kernel`、`ComfyUI-OmniXPU`、Docker 镜像）、[intel/sycl-tla](https://github.com/intel/sycl-tla)、
[xiangyuT/comfy-kitchen-xpu](https://github.com/xiangyuT/comfy-kitchen-xpu)、
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)、
[ComfyUI](https://github.com/comfyanonymous/ComfyUI)。这是一个独立项目，与它们均无隶属或背书关系。
