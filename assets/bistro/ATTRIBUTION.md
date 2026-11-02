# Amazon Lumberyard Bistro

来源：Amazon Lumberyard, Open Research Content Archive (ORCA), 2017。
https://developer.nvidia.com/orca/amazon-lumberyard-bistro

许可证：Creative Commons Attribution 4.0 International，完整条款见 LICENSE.txt。
https://creativecommons.org/licenses/by/4.0/

2026-11-02由Codex适配：选取外景局部街区，保留XY原坐标范围[-35,45]×[-30,42]内相交对象，整体缩放0.5；FBX转glTF、DDS转PNG、缩小纹理，DirectX法线翻转绿色并重建蓝色；按原包规定将Specular纹理G/B作为roughness/metallic。植被使用MASK，玻璃使用基础BLEND近似；添加独立移动木箱。不是原版引擎画面的复刻，不包含原场景GI或玻璃折射。

原包、转换脚本及记录存放于 E:/Desktop/招聘要求/ToyRenderer-records/20261102-bistro，正式渲染不依赖该目录。

2026-11-02后续修复：恢复裁剪遗漏的13个原始道路对象，并替换6231的错位导出；14对象按原始世界坐标缩放0.5烘焙顶点，材质沿用已有转换。转换与坐标验证记录在项目外20261102-floor-fix。
