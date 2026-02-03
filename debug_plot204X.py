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
data_dir = r"D:\movefromHP\研究生\新疆项目-超深层页岩油压裂缝网量化评价技术\水锤数据\夏204X原始数据\csv"
output_dir = os.path.join(data_dir, "最终诊断报告_表格版")
if not os.path.exists(output_dir): os.makedirs(output_dir)

stage_configs = {
    "第1段": [6051.0, 6037.0, 6024.5],
    "第2段": [6004.5, 5984.5, 5970.5, 5958.5],
    "第3段": [5938.0, 5928.0, 5914.0],
    "第4段": [5891.0, 5873.0, 5857.0],
    "第5段": [5836.0, 5825.5, 5815.0, 5806.0],
}

# 核心区间调节参数
que_buffer = 0.2

# 物理与算法参数
wave_speed = 1382.5
filter_low = 2.0
filter_high = 45.0
clip_time = 70
current_threshold = 0.0011
match_error_limit = 10.0

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 核心分析逻辑 (优化为全局最优匹配)
# ==========================================
def analyze_stage_with_table(file_path, target_depths):
    df = pd.read_csv(file_path, header=None)
    time, pressure = df.iloc[:, 0].values, df.iloc[:, 1].values
    if clip_time > 0 and time[-1] > clip_time:
        idx = np.argmax(time > clip_time)
        time, pressure = time[:idx], pressure[:idx]

    dt = np.median(np.diff(time))
    fs = 1.0 / dt
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

    t_min = (2 * min(target_depths)) / wave_speed - que_buffer
    t_max = (2 * max(target_depths)) / wave_speed + que_buffer

    mask = (que > t_min) & (que < t_max)
    v_que, v_cep = que[mask], cep[mask]
    peaks, _ = find_peaks(v_cep, prominence=current_threshold)

    # 识别出的所有潜在点
    detected_p = [{'depth': (wave_speed * v_que[p]) / 2, 'amp': v_cep[p], 'que': v_que[p]} for p in peaks]

    # --- 全局最优匹配算法：谁近归谁 ---
    res = []
    # 初始化结果列表
    for i, target in enumerate(target_depths):
        res.append({'cluster': f"第{i + 1}簇", 'target': target, 'status': '×',
                    'depth': "--", 'error': "--", 'que': None, 'amp': 0})

    # 构建所有可能的“理论-实测”配对情况
    all_pairs = []
    for i, target in enumerate(target_depths):
        for p_idx, p_data in enumerate(detected_p):
            error = abs(p_data['depth'] - target)
            if error < match_error_limit:
                all_pairs.append({'target_idx': i, 'peak_idx': p_idx, 'error': error, 'data': p_data})

    # 按误差从小到大排序，优先锁定距离最近的组合
    all_pairs.sort(key=lambda x: x['error'])

    assigned_targets = set()
    assigned_peaks = set()

    for pair in all_pairs:
        t_idx = pair['target_idx']
        p_idx = pair['peak_idx']
        # 如果这个射孔簇和这个实测点都还没被匹配过
        if t_idx not in assigned_targets and p_idx not in assigned_peaks:
            p_b = pair['data']
            res[t_idx].update({
                'status': '√',
                'depth': round(p_b['depth'], 2),
                'error': round(p_b['depth'] - target_depths[t_idx], 2),
                'que': p_b['que'],
                'amp': p_b['amp']
            })
            assigned_targets.add(t_idx)
            assigned_peaks.add(p_idx)

    return res, (v_que, v_cep)


# ==========================================
# 3. 绘图与仪表盘生成
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for stage_name, targets in stage_configs.items():
        f_path = next((f for f in all_files if stage_name in f), None)
        if not f_path: continue

        print(f"📊 正在生成 {stage_name} 的详细诊断图...")
        results, (v_que, v_cep) = analyze_stage_with_table(f_path, targets)
        open_rate = (sum(1 for r in results if r['status'] == '√') / len(targets)) * 100

        fig = plt.figure(figsize=(16, 11))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.8, 1], width_ratios=[3, 1], hspace=0.35)

        # --- A. 倒频谱波形 ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(v_que, v_cep, color='#1f77b4', alpha=0.6, linewidth=1.2, label='倒频谱信号')
        for r in results:
            t_theory = (2 * r['target']) / wave_speed
            ax0.axvline(x=t_theory, color='green', ls='--', alpha=0.3)
            ax0.text(t_theory, ax0.get_ylim()[1] * 0.96, r['cluster'], rotation=90, va='top', ha='right', color='green',
                     fontsize=12)
            if r['status'] == '√':
                ax0.plot(r['que'], r['amp'], "ro", ms=9, mec='w', mew=1.2)

        ax0.set_title(f"【{stage_name}】水锤反射深度诊断 (开启率: {open_rate:.1f}%)", fontsize=14, fontweight='bold', pad=15)
        ax0.set_xlabel("倒频率 (s)", fontsize=20)
        ax0.set_ylabel("相对幅值", fontsize=20)
        ax0.grid(ls=':', alpha=0.4)
        ax0.set_ylim(bottom=0)
        ax0.tick_params(axis='both', which='major', labelsize=20)

        # --- B. 开启状态对比 ---
        ax1 = fig.add_subplot(gs[0, 1])
        clrs = ['#2ca02c' if r['status'] == '√' else '#d62728' for r in results]
        bars = ax1.bar(range(len(targets)), [1] * len(targets), color=clrs, alpha=0.7, edgecolor='k', width=0.6)
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels([r['cluster'] for r in results], fontsize=20)
        ax1.set_yticks([])
        for i, bar in enumerate(bars):
            txt = "开启" if results[i]['status'] == '√' else "未识别"
            ax1.text(bar.get_x() + bar.get_width() / 2, 1.05, txt, ha='center', va='bottom',
                     fontweight='bold', color=clrs[i], fontsize=16)

        # --- C. 详细数据表格 ---
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        table_vals = [[r['cluster'], f"{r['target']}m", r['status'],
                       f"{r['depth']}m" if r['depth'] != "--" else "--",
                       f"{r['error']}m" if r['error'] != "--" else "--"] for r in results]

        cols = ["目标簇序号", "理论射孔深度", "开启状态评价", "实测反射深度", "计算误差"]
        diag_table = ax_table.table(cellText=table_vals, colLabels=cols, loc='center', cellLoc='center')

        diag_table.auto_set_font_size(False)
        diag_table.set_fontsize(20)  # 统一为20号字体
        diag_table.scale(1, 2.4)

        for (row, col), cell in diag_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            elif row > 0 and table_vals[row - 1][2] == '√':
                cell.set_facecolor('#f2fff2')

        plt.subplots_adjust(bottom=0.08, top=0.92)
        save_name = os.path.join(output_dir, f"{stage_name}_评价报告.png")
        plt.savefig(save_name, dpi=200, bbox_inches='tight')

        # 保持单张显示，方便调整大小
        plt.show()

    print(f"\n✨ 全部处理完成！报告已保存至: {output_dir}")
