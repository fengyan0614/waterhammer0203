import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, ifft

# ==========================================
# 1. 批量配置区域
# ==========================================
# 数据路径
data_dir = r"D:\movefromHP\研究生\新疆项目-超深层页岩油压裂缝网量化评价技术\水锤数据\Max5127原始数据局\新建文件夹"
output_dir = os.path.join(data_dir, "标定修正_最终诊断报告")
if not os.path.exists(output_dir): os.makedirs(output_dir)

# --- 标定配置：桥塞/井底深度 ---
calibration_depths = {
    "第4段": 6116.0, "第5-5段": 6060.5, "第11段": 5763.5, "第12段": 5713.0,
    "第13段": 5658.0, "第14段": 5611.0, "第15段": 5566.5, "第16段": 5521.5,
    "第17段": 5470.0, "第18段": 5420.0, "第19段": 5370.0, "第20段": 5319.0,
    "第21段": 5269.0,
}

# --- 诊断配置：射孔簇设计深度 ---
stage_configs = {
    "第4段": [6108.5, 6099.5, 6093.5, 6084.5, 6075.5, 6066.5],
    "第5-5段": [6055.0, 6042.5, 6031.0, 6017.5],
    "第11段": [5755.5, 5743.5, 5734.5, 5720.5],
    "第12段": [5703.5, 5690.5, 5681.0, 5664.0],
    "第13段": [5652.5, 5640.5, 5628.5, 5617.5],
    "第14段": [5606.0, 5595.0, 5585.5, 5573.5],
    "第15段": [5561.0, 5541.5, 5528.5],
    "第16段": [5516.0, 5508.0, 5499.0, 5487.5, 5482.0, 5476.0],
    "第17段": [5461.0, 5448.5, 5437.5, 5428.0],
    "第18段": [5414.5, 5404.0, 5393.0, 5380.5],
    "第19段": [5362.5, 5348.5, 5337.0, 5324.5],
    "第20段": [5313.5, 5302.0, 5284.5, 5274.0],
    "第21段": [5258.0, 5247.0, 5235.0, 5223.5],
}

# 物理与算法参数
initial_wave_speed = 1380.0  # 标定用初始值
filter_low, filter_high = 2.0, 45.0
clip_time = 70
current_threshold = 0.0011  # 显著性阈值
match_error_limit = 10.0  # 修改为 5.0m，对标光纤精度
que_buffer = 0.2  # 倒频谱显示窗口缓冲 (s)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 核心处理逻辑
# ==========================================

def get_signal_data(file_path):
    """读取并处理信号，返回倒频谱"""
    df = pd.read_csv(file_path, header=None)
    time, pressure = df.iloc[:, 0].values, df.iloc[:, 1].values
    if clip_time > 0 and time[-1] > clip_time:
        idx = np.argmax(time > clip_time)
        time, pressure = time[:idx], pressure[:idx]

    dt = np.median(np.diff(time))
    fs = 1.0 / dt

    # 滤波预处理
    p_med = np.median(pressure)
    p_mad = np.median(np.abs(pressure - p_med))
    pressure = np.clip(pressure, p_med - 3 * p_mad, p_med + 3 * p_mad)

    nyq = 0.5 * fs
    b, a = butter(5, [filter_low / nyq, filter_high / nyq], btype='band')
    p_filt = filtfilt(b, a, pressure)

    spec = fft(p_filt)
    log_spec = np.log(np.abs(spec) / np.max(np.abs(spec)) + 1e-10)
    cep = np.abs(ifft(log_spec))
    que = np.arange(len(cep)) / fs
    return que, cep, dt


def calibrate_single_file(que, cep, L_ref):
    """标定单个文件的波速"""
    t_theo = (2 * L_ref) / initial_wave_speed
    mask = (que > t_theo - 0.12) & (que < t_theo + 0.12)
    if not any(mask): return None
    t_ref = que[mask][np.argmax(cep[mask])]
    return (2 * L_ref) / t_ref


def analyze_with_calibrated_speed(que, cep, target_depths, used_speed):
    """基于修正波速的全局最优匹配"""
    # 自动确定寻峰区间
    t_min = (2 * min(target_depths)) / used_speed - que_buffer
    t_max = (2 * max(target_depths)) / used_speed + que_buffer
    mask = (que > t_min) & (que < t_max)
    v_que, v_cep = que[mask], cep[mask]

    peaks, _ = find_peaks(v_cep, prominence=current_threshold)
    detected_p = [{'depth': (used_speed * v_que[p]) / 2, 'amp': v_cep[p], 'que': v_que[p]} for p in peaks]

    res = []
    for i, target in enumerate(target_depths):
        res.append({'cluster': f"第{i + 1}簇", 'target': target, 'status': '×',
                    'depth': "--", 'error': "--", 'que': None, 'amp': 0})

    all_pairs = []
    for i, target in enumerate(target_depths):
        for p_idx, p_data in enumerate(detected_p):
            error = abs(p_data['depth'] - target)
            if error < match_error_limit:
                all_pairs.append({'target_idx': i, 'peak_idx': p_idx, 'error': error, 'data': p_data})

    all_pairs.sort(key=lambda x: x['error'])
    assigned_targets, assigned_peaks = set(), set()

    for pair in all_pairs:
        t_idx, p_idx = pair['target_idx'], pair['peak_idx']
        if t_idx not in assigned_targets and p_idx not in assigned_peaks:
            p_b = pair['data']
            res[t_idx].update({
                'status': '√', 'depth': round(p_b['depth'], 2),
                'error': round(p_b['depth'] - target_depths[t_idx], 2),
                'que': p_b['que'], 'amp': p_b['amp']
            })
            assigned_targets.add(t_idx)
            assigned_peaks.add(p_idx)

    return res, (v_que, v_cep)


# ==========================================
# 3. 执行引擎
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    # 阶段一修改：不再预计算平均波速，改为遍历文件时实时标定
    for stage_name, targets in stage_configs.items():
        f_path = next((f for f in all_files if stage_name in f), None)
        if not f_path: continue

        # 查找当前段的标定深度
        L_ref = next((d for k, d in calibration_depths.items() if k in stage_name), None)
        if not L_ref:
            print(f"⚠️ {stage_name} 未配置桥塞标定深度，跳过。")
            continue

        print(f"📊 正在处理 {stage_name} | 正在逐段标定波速...")
        que, cep, _ = get_signal_data(f_path)

        # 实时计算该段对应的波速
        current_speed = calibrate_single_file(que, cep, L_ref)

        if not current_speed:
            print(f"❌ {stage_name} 标定失败，无法识别桥塞回波。")
            continue

        print(f"  > 标定波速: {current_speed:.2f} m/s")

        # 使用当前段波速进行诊断
        results, (v_que, v_cep) = analyze_with_calibrated_speed(que, cep, targets, current_speed)

        open_rate = (sum(1 for r in results if r['status'] == '√') / len(targets)) * 100

        # --- 绘图逻辑 (保持不变) ---
        fig = plt.figure(figsize=(19, 14))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.8, 1], width_ratios=[3, 1], hspace=0.35)

        # A. 倒频谱图
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(v_que, v_cep, color='#1f77b4', alpha=0.6, lw=1.2)
        for r in results:
            t_theo = (2 * r['target']) / current_speed
            ax0.axvline(x=t_theo, color='green', ls='--', alpha=0.3)
            ax0.text(t_theo, ax0.get_ylim()[1] * 0.95, r['cluster'], rotation=90, color='green', ha='right')
            if r['status'] == '√':
                ax0.plot(r['que'], r['amp'], "ro", ms=8)

        ax0.set_title(f"【{stage_name}】深度诊断 (逐段修正波速: {current_speed:.1f} m/s | 开启率: {open_rate:.1f}%)",
                      fontsize=14, fontweight='bold')
        ax0.set_xlabel("倒频率 (s)", fontsize=16)
        ax0.set_ylabel("相对幅值", fontsize=16)
        ax0.tick_params(axis='both', which='major', labelsize=16)
        # B. 开启状态柱状图
        ax1 = fig.add_subplot(gs[0, 1])
        clrs = ['#2ca02c' if r['status'] == '√' else '#d62728' for r in results]
        ax1.bar(range(len(targets)), [1] * len(targets), color=clrs, alpha=0.7)
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels([r['cluster'] for r in results])
        ax1.set_yticks([])
        ax1.tick_params(axis='both', which='major', labelsize=14)
        # C. 详细数据表格
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')
        table_vals = [[r['cluster'], f"{r['target']}m", r['status'],
                       f"{r['depth']}m" if r['depth'] != "--" else "--",
                       f"{r['error']}m" if r['error'] != "--" else "--"] for r in results]

        diag_table = ax_table.table(cellText=table_vals, colLabels=["簇号", "设计深度", "状态", "实测深度", "误差"], loc='center',
                                    cellLoc='center')
        diag_table.auto_set_font_size(False)
        diag_table.set_fontsize(14)
        diag_table.scale(1, 2.2)

        plt.savefig(os.path.join(output_dir, f"{stage_name}_逐段修正报告.png"), dpi=200, bbox_inches='tight')
        plt.show()

    print(f"\n✨ 全部逐段标定诊断完成！结果目录: {output_dir}")
