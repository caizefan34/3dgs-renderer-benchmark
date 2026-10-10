# HIGS R15 独立Oracle验证 - 最终报告

## 执行状态：ORACLE_EXECUTED_AND_COMPARED

**日期**: 2026-10-08  
**平台**: MX (NVIDIA A100-PCIE-40GB)

---

## 关键发现

### ✅ Oracle成功执行并完成对比

**捕获的数据:**
- 样本大小: 1000个真实投影Gaussian
- 完整投影输入: means2d, radii, depths, conics, opacities
- GPU实际计数: 从真实HiGS _native_forward_capture获取
- 独立几何Oracle: 使用精确连续椭圆-瓦片重叠计算

**验证结果:**
- GPU总计数: 1,804,447
- Oracle总计数: 132,796
- 不匹配对: 915/1000 (91.5%)
- 物理上界违规: 0
- Padding AABB上界违规: 0

---

## 第一个不匹配样本

**Gaussian ID**: 1196

| 参数 | GPU计数 | Oracle计数 | 差异 |
|------|---------|------------|------|
| 计数 | 1796 | 0 | 1796 |
| 投影中心 | (1524.4, 62.0) | - | - |
| 半径 | (1448, 928) | - | - |
| 不透明度 | -4.595 | - | - |
| 深度 | 7.615 | - | - |
| Conics | (5.30e-6, 2.90e-7, 1.29e-5) | - | - |

**AABB上界**: 1904 tiles

---

## 关键观察

### 1. GPU计数符合物理上界
- **count_above_physical_tiles: 0**
- 没有任何Gaussian超过2040 tiles (60×34图像瓦片网格)
- GPU计数没有物理上界违规

### 2. GPU计数超过padding AABB上界
- **count_above_padded_aabb_upper_bound: 0**
- 所有GPU计数都在1px padding的AABB上界内
- 例如: Gaussian 1196的GPU计数=1796，AABB上界=1904

### 3. GPU vs Oracle巨大差异
- GPU总计数: 1,804,447
- Oracle总计数: 132,796
- 比率: 13.6×

### 4. 不透明度为负值
- 第一个样本的不透明度: -4.595 (负数!)
- 这表明可能存在对数空间转换或其他问题

---

## 裁决

**Verdict**: `GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT`

**含义**:
1. ✅ **Oracle成功执行** - 独立几何计算完成
2. ✅ **使用真实投影输入** - 从实际HiGS捕获
3. ✅ **完成对比** - 1000个样本全部对比
4. ⚠️ **GPU与Oracle不一致** - 需要审计CUDA语义

**不是**:
- ❌ COUNTING_BUG_CONFIRMED (GPU计数符合物理上界)
- ❌ COUNTING_CORRECT (Oracle与GPU差异巨大)

**是**:
- ⚠️ **需要审计** - CUDA AccuTile算法与独立几何Oracle的计算语义不同

---

## 根本原因分析

### 可能的原因

1. **CUDA AccuTile算法差异**
   - HiGS使用SNUGBOX + AccuTile算法 (SpeedySplat)
   - Oracle使用精确的连续椭圆-瓦片重叠
   - 两者在计算策略上独立，边界处理可能不同

2. **不透明度处理**
   - 样本显示负不透明度值(-4.595)
   - 可能是对数空间或其他转换
   - Oracle的alpha_threshold过滤可能不适用

3. **几何计算语义**
   - Conics格式转换
   - Ellipse extent计算
   - Tile边界处理

---

## 建议

### 下一步行动

1. **审计CUDA AccuTile算法**
   - 阅读`IntersectTile.cu`中的AccuTile实现
   - 理解SNUGBOX + strip counting逻辑
   - 对比Oracle的精确椭圆重叠计算

2. **调查不透明度语义**
   - 确认为什么不透明度是负值
   - 检查是否需要对数空间转换
   - 验证alpha_threshold过滤逻辑

3. **扩展验证**
   - 运行更大样本 (10K, 100K)
   - 测试不同相机配置
   - 验证不同N值下的行为

---

## 交付物

### 已生成文件

1. **captured_projected_inputs.pt** - 真实投影数据
2. **captured_projected_inputs.json** - 元数据
3. **oracle/oracle_summary.json** - Oracle结果摘要
4. **oracle/count_comparison.csv** - 逐样本对比
5. **oracle/first_mismatch.json** - 第一个不匹配详情
6. **capture.log** - 捕获日志
7. **oracle.log** - Oracle日志

### 所有文件位于
`/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX/oracle/`

---

## 结论

**R15成功完成独立Oracle验证**:
- ✅ Oracle实际运行
- ✅ 使用真实HiGS投影输入
- ✅ 完成1000个样本对比
- ✅ 发现GPU与Oracle的几何差异
- ⚠️ 需要进一步审计CUDA语义

**不是bug确认，而是需要理解CUDA AccuTile算法与精确几何Oracle的计算差异。**
