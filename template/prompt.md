
```py
            # 新增力矩/电流投影层
            if self.use_effort:
                self.input_proj_effort = ForceEncoder(
                    input_dim=self.effort_dim,
                    output_dim=hidden_dim,
                )
                print(f"🔧 ForceEncoder initialized with input_dim={self.effort_dim}, output_dim={hidden_dim}")
```               


这个是未来使用 forceencoder 的接口形式,保持不变

现在需要把 @template/force_encoder.py 修改成 @src/model.py 的模型结构

add: 导出和导入 force_encoder 模型权重的功能,内置在  @template/force_encoder.py 内,