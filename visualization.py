"""
시각화 모듈 (Enhanced Version)
- P-V 다이어그램
- 3D P-V-T 다이어그램 (Plotly)
- 애니메이션
- 열역학 사이클 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from thermodynamics import calculate_temperature, R, n, GAS_TYPES


# ==================== Matplotlib 기반 시각화 ====================

def plot_pv_diagram(paths: List[Dict], optimal_path: Optional[Dict] = None,
                   P1: float = 5, V1: float = 2, P2: float = 1, V2: float = 8,
                   show_isotherms: bool = True, figsize: Tuple = (10, 8),
                   dark_mode: bool = False) -> Tuple:
    """P-V 다이어그램 그리기 (Matplotlib)"""

    # 다크모드 설정
    if dark_mode:
        plt.style.use('dark_background')
        bg_color = '#1e1e1e'
        text_color = 'white'
        grid_color = 'gray'
    else:
        plt.style.use('default')
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'lightgray'

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 등온선 표시
    if show_isotherms:
        T_values = [200, 300, 400, 500, 600, 700, 800]
        V_iso = np.linspace(0.5, 10.5, 200)
        for T in T_values:
            P_iso = (n * R * T) / V_iso
            mask = (P_iso >= 0.5) & (P_iso <= 10.5)
            ax.plot(V_iso[mask], P_iso[mask], color=grid_color, alpha=0.3,
                   linestyle='--', linewidth=0.8)
            # 온도 라벨
            if np.any(mask):
                idx = np.where(mask)[0][-1]
                if idx < len(V_iso) - 1:
                    ax.text(V_iso[idx], P_iso[idx], f'{T}K',
                           fontsize=8, color=grid_color, alpha=0.6)

    # 초기/최종 상태 점 표시
    ax.scatter([V1], [P1], s=300, c='#00ff88', marker='o',
              label='A (시작)', zorder=5, edgecolors='white', linewidths=2)
    ax.scatter([V2], [P2], s=300, c='#ff4444', marker='s',
              label='B (끝)', zorder=5, edgecolors='white', linewidths=2)

    # 온도 표시
    T1 = calculate_temperature(P1, V1)
    T2 = calculate_temperature(P2, V2)
    ax.text(V1, P1 + 0.4, f'T={T1:.0f}K', fontsize=11,
           ha='center', color='#00ff88', weight='bold')
    ax.text(V2, P2 + 0.4, f'T={T2:.0f}K', fontsize=11,
           ha='center', color='#ff4444', weight='bold')

    # 일반 경로들 표시
    colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#e91e63', '#00bcd4']
    for i, path in enumerate(paths):
        if 'P_array' in path and 'V_array' in path:
            color = colors[i % len(colors)]
            ax.plot(path['V_array'], path['P_array'], color=color,
                   linewidth=2.5, label=f"경로 {i+1}: {path.get('type', '일반')}",
                   alpha=0.85, zorder=3)
            # 방향 화살표
            mid = len(path['V_array']) // 2
            if mid > 0:
                ax.annotate('', xy=(path['V_array'][mid+1], path['P_array'][mid+1]),
                           xytext=(path['V_array'][mid], path['P_array'][mid]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # 최적 경로 표시
    if optimal_path and 'P_array' in optimal_path and 'V_array' in optimal_path:
        ax.plot(optimal_path['V_array'], optimal_path['P_array'],
               color='#ff0066', linewidth=4, label='⭐ 최적 경로',
               zorder=4, alpha=0.95)
        # 영역 채우기 (일 시각화)
        ax.fill_between(optimal_path['V_array'], optimal_path['P_array'],
                        alpha=0.15, color='#ff0066')

    # 축 설정
    ax.set_xlabel("부피 V (L)", fontsize=14, weight='bold', color=text_color)
    ax.set_ylabel("압력 P (atm)", fontsize=14, weight='bold', color=text_color)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', color=grid_color)
    ax.set_title("P-V 다이어그램", fontsize=16, weight='bold', pad=15, color=text_color)
    ax.tick_params(colors=text_color)

    plt.tight_layout()
    return fig, ax


def plot_work_comparison(paths: List[Dict], optimal_path: Optional[Dict] = None,
                        W_reversible: Optional[float] = None,
                        figsize: Tuple = (10, 5), dark_mode: bool = False) -> Tuple:
    """경로별 일 비교 막대 그래프"""

    if dark_mode:
        plt.style.use('dark_background')
        bg_color = '#1e1e1e'
    else:
        plt.style.use('default')
        bg_color = 'white'

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    names = []
    works = []
    colors_list = []

    for i, path in enumerate(paths):
        if 'W' in path:
            names.append(f"경로 {i+1}\n({path.get('type', '일반')})")
            works.append(path['W'])
            colors_list.append('#3498db')

    if optimal_path and 'W' in optimal_path:
        names.append("⭐ 최적 경로")
        works.append(optimal_path['W'])
        colors_list.append('#ff0066')

    if len(works) == 0:
        ax.text(0.5, 0.5, '경로를 추가해주세요',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig, ax

    bars = ax.bar(names, works, color=colors_list, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    if W_reversible is not None:
        ax.axhline(y=W_reversible, color='#00ff88', linestyle='--',
                  linewidth=2.5, label=f'가역 과정 ({W_reversible:.2f} L·atm)')
        ax.legend(fontsize=11)

    for bar, work in zip(bars, works):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{work:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_ylabel("일 W (L·atm)", fontsize=13, weight='bold')
    ax.set_title("경로별 한 일 비교", fontsize=15, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig, ax


def plot_efficiency_comparison(paths: List[Dict], optimal_path: Optional[Dict] = None,
                              figsize: Tuple = (10, 5), dark_mode: bool = False) -> Tuple:
    """경로별 효율 비교 막대 그래프"""

    if dark_mode:
        plt.style.use('dark_background')
        bg_color = '#1e1e1e'
    else:
        plt.style.use('default')
        bg_color = 'white'

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    names = []
    efficiencies = []
    colors_list = []

    for i, path in enumerate(paths):
        if 'efficiency' in path:
            names.append(f"경로 {i+1}\n({path.get('type', '일반')})")
            efficiencies.append(path['efficiency'])
            colors_list.append('#3498db')

    if optimal_path and 'efficiency' in optimal_path:
        names.append("⭐ 최적 경로")
        efficiencies.append(optimal_path['efficiency'])
        colors_list.append('#ff0066')

    if len(efficiencies) == 0:
        ax.text(0.5, 0.5, '경로를 추가해주세요',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig, ax

    bars = ax.bar(names, efficiencies, color=colors_list, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    ax.axhline(y=100, color='#00ff88', linestyle='--', linewidth=2,
              label='100% (가역 과정)')
    ax.legend(fontsize=11)

    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{eff:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_ylabel("효율 (%)", fontsize=13, weight='bold')
    ax.set_title("경로별 효율 비교", fontsize=15, weight='bold', pad=15)
    ax.set_ylim(0, max(110, max(efficiencies) * 1.1) if efficiencies else 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig, ax


# ==================== Plotly 기반 3D 시각화 ====================

def plot_3d_pvt_diagram(paths: List[Dict], optimal_path: Optional[Dict] = None,
                       P1: float = 5, V1: float = 2, P2: float = 1, V2: float = 8,
                       show_surface: bool = True, dark_mode: bool = True) -> go.Figure:
    """3D P-V-T 다이어그램 (Plotly)"""

    # 테마 설정
    if dark_mode:
        template = 'plotly_dark'
        bg_color = '#1e1e1e'
        grid_color = '#444444'
    else:
        template = 'plotly_white'
        bg_color = 'white'
        grid_color = '#cccccc'

    fig = go.Figure()

    # 이상기체 상태방정식 표면
    if show_surface:
        V_surf = np.linspace(1, 10, 30)
        T_surf = np.linspace(200, 800, 30)
        V_mesh, T_mesh = np.meshgrid(V_surf, T_surf)
        P_mesh = (n * R * T_mesh) / V_mesh

        fig.add_trace(go.Surface(
            x=V_mesh, y=T_mesh, z=P_mesh,
            colorscale='Viridis',
            opacity=0.4,
            showscale=False,
            name='상태방정식 표면'
        ))

    # 초기/최종 상태점
    T1 = calculate_temperature(P1, V1)
    T2 = calculate_temperature(P2, V2)

    fig.add_trace(go.Scatter3d(
        x=[V1], y=[T1], z=[P1],
        mode='markers+text',
        marker=dict(size=12, color='#00ff88', symbol='circle'),
        text=['A (시작)'],
        textposition='top center',
        name='초기 상태 A'
    ))

    fig.add_trace(go.Scatter3d(
        x=[V2], y=[T2], z=[P2],
        mode='markers+text',
        marker=dict(size=12, color='#ff4444', symbol='square'),
        text=['B (끝)'],
        textposition='top center',
        name='최종 상태 B'
    ))

    # 일반 경로들
    colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#e91e63']
    for i, path in enumerate(paths):
        if 'P_array' in path and 'V_array' in path:
            T_array = np.array([calculate_temperature(p, v)
                               for p, v in zip(path['P_array'], path['V_array'])])
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter3d(
                x=path['V_array'],
                y=T_array,
                z=path['P_array'],
                mode='lines',
                line=dict(color=color, width=6),
                name=f"경로 {i+1}: {path.get('type', '일반')}"
            ))

    # 최적 경로
    if optimal_path and 'P_array' in optimal_path and 'V_array' in optimal_path:
        T_opt = np.array([calculate_temperature(p, v)
                        for p, v in zip(optimal_path['P_array'], optimal_path['V_array'])])

        fig.add_trace(go.Scatter3d(
            x=optimal_path['V_array'],
            y=T_opt,
            z=optimal_path['P_array'],
            mode='lines',
            line=dict(color='#ff0066', width=8),
            name='⭐ 최적 경로'
        ))

    # 레이아웃 설정
    fig.update_layout(
        template=template,
        title=dict(
            text='3D P-V-T 다이어그램',
            font=dict(size=20, color='white' if dark_mode else 'black')
        ),
        scene=dict(
            xaxis=dict(title='부피 V (L)', gridcolor=grid_color),
            yaxis=dict(title='온도 T (K)', gridcolor=grid_color),
            zaxis=dict(title='압력 P (atm)', gridcolor=grid_color),
            bgcolor=bg_color
        ),
        paper_bgcolor=bg_color,
        legend=dict(
            font=dict(size=12),
            bgcolor='rgba(0,0,0,0.5)' if dark_mode else 'rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def plot_cycle_diagram(cycle_data: Dict, dark_mode: bool = True) -> go.Figure:
    """열역학 사이클 다이어그램 (Plotly)"""

    if dark_mode:
        template = 'plotly_dark'
        bg_color = '#1e1e1e'
    else:
        template = 'plotly_white'
        bg_color = 'white'

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P-V 다이어그램', 'T-S 다이어그램'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    paths = cycle_data.get('paths', [])

    # P-V 다이어그램
    for i, path in enumerate(paths):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=path['V'],
                y=path['P'],
                mode='lines',
                name=f"{path.get('step', '')}: {path.get('type', '')}",
                line=dict(color=color, width=3),
                showlegend=True
            ),
            row=1, col=1
        )

    # 상태점 표시
    states = cycle_data.get('states', {})
    for state_name, state in states.items():
        fig.add_trace(
            go.Scatter(
                x=[state['V']],
                y=[state['P']],
                mode='markers+text',
                marker=dict(size=12, color='white', line=dict(width=2, color='black')),
                text=[state_name],
                textposition='top center',
                showlegend=False
            ),
            row=1, col=1
        )

    # 사이클 정보
    cycle_type = cycle_data.get('cycle_type', 'Unknown')
    efficiency = cycle_data.get('efficiency', 0)
    W_net = cycle_data.get('W_net', 0)

    fig.update_layout(
        template=template,
        title=dict(
            text=f'{cycle_type} 사이클 | 효율: {efficiency:.1f}% | 순일: {W_net:.2f} L·atm',
            font=dict(size=18)
        ),
        paper_bgcolor=bg_color,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.update_xaxes(title_text='부피 V (L)', row=1, col=1)
    fig.update_yaxes(title_text='압력 P (atm)', row=1, col=1)

    return fig


def plot_animated_path(paths: List[Dict], P1: float, V1: float,
                      P2: float, V2: float, dark_mode: bool = True) -> go.Figure:
    """경로 애니메이션 (Plotly)"""

    if dark_mode:
        template = 'plotly_dark'
        bg_color = '#1e1e1e'
    else:
        template = 'plotly_white'
        bg_color = 'white'

    fig = go.Figure()

    # 등온선 배경
    T_values = [300, 400, 500, 600, 700]
    V_iso = np.linspace(0.5, 10.5, 100)
    for T in T_values:
        P_iso = (n * R * T) / V_iso
        mask = (P_iso >= 0.5) & (P_iso <= 10.5)
        fig.add_trace(go.Scatter(
            x=V_iso[mask], y=P_iso[mask],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.3,
            showlegend=False
        ))

    # 초기/최종 상태점
    fig.add_trace(go.Scatter(
        x=[V1], y=[P1],
        mode='markers+text',
        marker=dict(size=20, color='#00ff88'),
        text=['A'],
        textposition='top center',
        name='시작점 A'
    ))

    fig.add_trace(go.Scatter(
        x=[V2], y=[P2],
        mode='markers+text',
        marker=dict(size=20, color='#ff4444'),
        text=['B'],
        textposition='top center',
        name='끝점 B'
    ))

    # 애니메이션 프레임 생성
    frames = []
    colors = ['#3498db', '#e67e22', '#9b59b6', '#ff0066']

    for path_idx, path in enumerate(paths):
        if 'P_array' not in path or 'V_array' not in path:
            continue

        P_array = path['P_array']
        V_array = path['V_array']
        color = colors[path_idx % len(colors)]

        for i in range(2, len(P_array) + 1):
            frame_data = []

            # 이전 경로들 (완성된 것)
            for prev_idx in range(path_idx):
                prev_path = paths[prev_idx]
                if 'P_array' in prev_path and 'V_array' in prev_path:
                    frame_data.append(go.Scatter(
                        x=prev_path['V_array'],
                        y=prev_path['P_array'],
                        mode='lines',
                        line=dict(color=colors[prev_idx % len(colors)], width=3),
                        opacity=0.5
                    ))

            # 현재 경로 (진행 중)
            frame_data.append(go.Scatter(
                x=V_array[:i],
                y=P_array[:i],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color)
            ))

            # 현재 위치 마커
            frame_data.append(go.Scatter(
                x=[V_array[i-1]],
                y=[P_array[i-1]],
                mode='markers',
                marker=dict(size=15, color='yellow', symbol='star')
            ))

            frames.append(go.Frame(
                data=frame_data,
                name=f'{path_idx}_{i}'
            ))

    fig.frames = frames

    # 애니메이션 버튼
    fig.update_layout(
        template=template,
        title='경로 애니메이션',
        xaxis=dict(title='부피 V (L)', range=[0.5, 10.5]),
        yaxis=dict(title='압력 P (atm)', range=[0.5, 10.5]),
        paper_bgcolor=bg_color,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ 재생',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏸ 일시정지',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'currentvalue': {'prefix': '프레임: '},
            'steps': [
                {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                      'mode': 'immediate'}],
                 'label': f.name,
                 'method': 'animate'}
                for f in frames
            ]
        }] if frames else []
    )

    return fig


def plot_thermodynamic_properties(paths: List[Dict], optimal_path: Optional[Dict] = None,
                                  dark_mode: bool = True) -> go.Figure:
    """열역학적 성질 종합 비교 차트"""

    if dark_mode:
        template = 'plotly_dark'
        bg_color = '#1e1e1e'
    else:
        template = 'plotly_white'
        bg_color = 'white'

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('일 (W)', '열 (Q)', '엔트로피 변화 (ΔS)', '효율 (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    all_paths = paths.copy()
    if optimal_path:
        all_paths.append(optimal_path)

    names = []
    W_values = []
    Q_values = []
    dS_values = []
    eff_values = []
    colors = []

    for i, path in enumerate(all_paths):
        if path == optimal_path:
            names.append('⭐ 최적')
            colors.append('#ff0066')
        else:
            names.append(f"경로 {i+1}")
            colors.append('#3498db')

        W_values.append(path.get('W', 0))
        Q_values.append(path.get('Q', 0))
        dS_values.append(path.get('dS', 0))
        eff_values.append(path.get('efficiency', 0))

    # 일 (W)
    fig.add_trace(
        go.Bar(x=names, y=W_values, marker_color=colors, name='일'),
        row=1, col=1
    )

    # 열 (Q)
    fig.add_trace(
        go.Bar(x=names, y=Q_values, marker_color=colors, name='열'),
        row=1, col=2
    )

    # 엔트로피 변화 (ΔS)
    fig.add_trace(
        go.Bar(x=names, y=dS_values, marker_color=colors, name='ΔS'),
        row=2, col=1
    )

    # 효율 (%)
    fig.add_trace(
        go.Bar(x=names, y=eff_values, marker_color=colors, name='효율'),
        row=2, col=2
    )

    fig.update_layout(
        template=template,
        title='열역학적 성질 종합 비교',
        paper_bgcolor=bg_color,
        showlegend=False,
        height=600
    )

    fig.update_yaxes(title_text='L·atm', row=1, col=1)
    fig.update_yaxes(title_text='L·atm', row=1, col=2)
    fig.update_yaxes(title_text='L·atm/K', row=2, col=1)
    fig.update_yaxes(title_text='%', row=2, col=2)

    return fig


def plot_algorithm_comparison(comparison_results: Dict, dark_mode: bool = True) -> go.Figure:
    """알고리즘 비교 차트"""

    if dark_mode:
        template = 'plotly_dark'
        bg_color = '#1e1e1e'
    else:
        template = 'plotly_white'
        bg_color = 'white'

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('계산 시간 (초)', '찾은 일 (L·atm)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    names = []
    times = []
    works = []

    for algo_name, data in comparison_results.items():
        names.append(algo_name.replace('_', '\n'))
        times.append(data.get('time', 0))
        if data.get('result'):
            works.append(data['result'].get('W', 0))
        else:
            works.append(0)

    # 시간 비교
    fig.add_trace(
        go.Bar(x=names, y=times, marker_color='#3498db', name='시간'),
        row=1, col=1
    )

    # 일 비교
    fig.add_trace(
        go.Bar(x=names, y=works, marker_color='#e67e22', name='일'),
        row=1, col=2
    )

    fig.update_layout(
        template=template,
        title='알고리즘 성능 비교',
        paper_bgcolor=bg_color,
        showlegend=False
    )

    return fig


def create_export_data(paths: List[Dict], optimal_path: Optional[Dict] = None) -> str:
    """결과를 CSV 형식으로 내보내기"""
    import io

    output = io.StringIO()
    output.write("경로,타입,일(W),열(Q),ΔU,ΔH,ΔS,ΔG,효율(%)\n")

    for i, path in enumerate(paths):
        output.write(f"경로 {i+1},{path.get('type', '일반')},")
        output.write(f"{path.get('W', 0):.4f},")
        output.write(f"{path.get('Q', 0):.4f},")
        output.write(f"{path.get('dU', 0):.4f},")
        output.write(f"{path.get('dH', 0):.4f},")
        output.write(f"{path.get('dS', 0):.6f},")
        output.write(f"{path.get('dG', 0):.4f},")
        output.write(f"{path.get('efficiency', 0):.2f}\n")

    if optimal_path:
        output.write(f"최적 경로,{optimal_path.get('type', '최적')},")
        output.write(f"{optimal_path.get('W', 0):.4f},")
        output.write(f"{optimal_path.get('Q', 0):.4f},")
        output.write(f"{optimal_path.get('dU', 0):.4f},")
        output.write(f"{optimal_path.get('dH', 0):.4f},")
        output.write(f"{optimal_path.get('dS', 0):.6f},")
        output.write(f"{optimal_path.get('dG', 0):.4f},")
        output.write(f"{optimal_path.get('efficiency', 0):.2f}\n")

    return output.getvalue()
