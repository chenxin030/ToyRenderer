# ToyRenderer

Windows / C++20 / Vulkan 实时渲染项目：延迟 PBR/IBL、四级稳定 CSM、SSAO 与保边重建、刚体运动矢量与 TAA、Compute Tiled Deferred。

## 构建与运行

```powershell
cmake -S . -B ../build-ToyRenderer
cmake --build ../build-ToyRenderer --config Release
```

双击 `../build-ToyRenderer/Release/ToyRenderer.exe` 即可进入最终 Bistro 商业街场景。资产来源与适配见[资产署名](assets/bistro/ATTRIBUTION.md)。

右键拖动调整视角，滚轮缩放。面板提供：

- 720p / 1080p 窗口尺寸、Balanced / Quality / Baseline 效果预设。
- Camera tour 自动游览、木箱动画、白天/黄昏照明。
- 阴影、AO、TAA、分块光照和材质组件对比，以及 TAA 历史重置。
- GPU 逐 Pass 耗时、CPU submit 耗时和灯光剔除统计。

Balanced 使用半分辨率 16 样本 AO，Quality 使用全分辨率 32 样本 AO，均启用 TAA 和分块光照；Baseline 关闭 AO/TAA 和分块光照。阴影统一使用 1024 分辨率的四级 CSM。

Release 默认不请求验证层；Debug 默认启用可用的验证层和同步验证。