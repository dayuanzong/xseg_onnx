# XSeg ONNX 导出脚本

用于将 DFL XSeg 的 .npy 权重导出为 ONNX，并自动修复 ConvTranspose 的非对称 padding，避免 CUDA EP 回退到 CPU。

## 依赖
- Python 3.10
- tf2onnx
- onnx
- onnxruntime
- OpenCV
- DFL 内置模块（core.leras / facelib）

## 使用方式
在目录内运行：

```bash
python export_xseg_onnx.py
```

按提示输入：
- 模型目录路径（包含 XSeg_*.npy）
- 输出 ONNX 路径
- 用于一致性测试的切脸图片路径（可留空）

## 说明
- 导出后会进行一致性验证，输出 max_diff / mean_diff。
- 会将 ConvTranspose pads 从 [0,0,1,1] 调整为 [0,0,0,0]，并插入 Slice 保持输出一致。
