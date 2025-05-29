import torch
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import from_smiles
from torch_geometric.nn import AttentiveFP, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional

# 设置页面配置
st.set_page_config(
    page_title="Chemical Shift Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .plot-controls {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }

    .success-alert {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .warning-alert {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .info-alert {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }

    .compound-title {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

    .prediction-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }

    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# Load and cache the database
@st.cache_data
def load_database():
    """Load the phosphorus database"""
    try:
        df = pd.read_csv("Resources/phosphorus_O5.csv")
        if 'cansmi' in df.columns:
            df['smiles'] = df['cansmi']
        return df
    except FileNotFoundError:
        st.error("数据库文件 'Resources/phosphorus_O5.csv' 未找到")
        return None
    except Exception as e:
        st.error(f"加载数据库时出错: {str(e)}")
        return None


def search_database(smiles_list, database):
    """Search for SMILES in the database"""
    if database is None:
        return {}

    found_results = {}
    for smiles in smiles_list:
        matches = database[database['cansmi'] == smiles]
        if not matches.empty:
            match = matches.iloc[0]
            found_results[smiles] = {
                'shift': match['shift'],
                'solvent': match.get('LM', 'Unknown'),
                'num': match.get('num', ''),
                'MW': match.get('MW', ''),
                'envlab': match.get('envlab', ''),
                'P_oxidation_state': match.get('P_oxidation_state', ''),
                'P_valence': match.get('P_valence', '')
            }
    return found_results


# Load the models
@st.cache_resource
def load_attentivefp_model():
    model = AttentiveFP(
        in_channels=9,
        hidden_channels=66,
        out_channels=1,
        edge_dim=3,
        num_layers=5,
        num_timesteps=3,
        dropout=0.0446259777448801
    )
    model.load_state_dict(
        torch.load("Resources/AttentiveFP_model.pt", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model


class NMRShiftModel(torch.nn.Module):
    def __init__(
            self,
            in_channels=9,
            hidden_channels=185,
            num_layers=4,
            out_channels=1,
            dropout=0.11080081715730111,
            pooling_method='max',
            act='relu',
            jk='cat'
    ):
        super().__init__()
        from torch_geometric.nn import GCN
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            norm='batch_norm',
            jk=jk,
            act=act
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.pooling_method = pooling_method

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index)
        if self.pooling_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_method == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling_method == 'max':
            x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


@st.cache_resource
def load_gcn_model():
    model = NMRShiftModel(
        in_channels=9,
        hidden_channels=185,
        num_layers=4,
        out_channels=1,
        dropout=0.11080081715730111,
        pooling_method='max',
        act='relu',
        jk='cat'
    )
    model.load_state_dict(
        torch.load("Resources/GCN_model.pt", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model


def inverse_transform(scaled_value):
    center = 23.47
    scale = 22.69999886
    return scaled_value * scale + center


def predict_chemical_shift_single_model(smiles_list, model_type='attentivefp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'attentivefp':
        model = load_attentivefp_model().to(device)
    else:
        model = load_gcn_model().to(device)

    predictions = []
    raw_predictions = []
    for smile in smiles_list:
        try:
            g = from_smiles(smile)
            g.x = g.x.float()
            g = g.to(device)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
            with torch.no_grad():
                if model_type == 'attentivefp':
                    pred = model(g.x, g.edge_index, g.edge_attr, g.batch)
                else:
                    pred = model(g.x, g.edge_index, g.batch)
            raw_pred = pred.item()
            raw_predictions.append(raw_pred)
            actual_pred = inverse_transform(raw_pred)
            predictions.append(actual_pred)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
            raw_predictions.append(None)
    return predictions, raw_predictions


def predict_with_multiple_models(smiles_list, selected_models=['attentivefp'], database=None):
    results = []
    db_results = search_database(smiles_list, database)
    found_smiles = list(db_results.keys())
    not_found_smiles = [s for s in smiles_list if s not in found_smiles]
    ml_predictions = {}

    for model_type in selected_models:
        if not_found_smiles:
            predictions, raw_predictions = predict_chemical_shift_single_model(not_found_smiles, model_type)
            ml_predictions[model_type] = {
                'predictions': dict(zip(not_found_smiles, predictions)),
                'raw_predictions': dict(zip(not_found_smiles, raw_predictions))
            }

    for smiles in smiles_list:
        if smiles in db_results:
            db_data = db_results[smiles]
            result = {
                'smiles': smiles,
                'database': db_data['shift'],
                'source': 'database',
                'solvent': db_data['solvent'],
                'additional_info': {
                    'num': db_data['num'],
                    'MW': db_data['MW'],
                    'envlab': db_data['envlab'],
                    'P_oxidation_state': db_data['P_oxidation_state'],
                    'P_valence': db_data['P_valence']
                }
            }
            for model_type in selected_models:
                result[f'{model_type}_prediction'] = None
                result[f'{model_type}_raw'] = None
        else:
            result = {
                'smiles': smiles,
                'database': None,
                'source': 'ml_models',
                'solvent': 'N/A',
                'additional_info': {}
            }
            for model_type in selected_models:
                if model_type in ml_predictions:
                    result[f'{model_type}_prediction'] = ml_predictions[model_type]['predictions'].get(smiles, "Error")
                    result[f'{model_type}_raw'] = ml_predictions[model_type]['raw_predictions'].get(smiles, None)
                else:
                    result[f'{model_type}_prediction'] = None
                    result[f'{model_type}_raw'] = None
        results.append(result)
    return results


def create_enhanced_nmr_plot(results: List[Dict], selected_models: List[str],
                             use_database: bool = False, plot_config: Dict = None) -> go.Figure:
    """创建增强的NMR谱图，优化可读性和交互性"""
    import plotly.graph_objects as go
    import numpy as np

    if plot_config is None:
        plot_config = {
            'peak_width': 0.04,
            'show_annotations': True,
            'plot_style': 'professional',
            'color_scheme': 'default'
        }

    # 收集所有有效的化学位移数据
    all_shifts = []
    plot_data = []

    # 统一的颜色方案 - 与对比图保持一致
    colors = {
        'attentivefp': '#1f77b4',  # 明亮蓝色
        'gcn': '#ff7f0e',         # 明亮橙色
        'database': '#2ca02c'      # 明亮绿色
    }

    for i, result in enumerate(results):
        compound_id = f"化合物 {i + 1}"

        # 数据库数据
        if use_database and result.get('database') is not None:
            try:
                db_shift = float(result['database'])
                if -200 <= db_shift <= 200:
                    all_shifts.append(db_shift)
                    plot_data.append({
                        'compound': compound_id,
                        'shift': db_shift,
                        'type': '数据库实验值',
                        'model': 'database',
                        'smiles': result['smiles'],
                        'solvent': result.get('solvent', 'N/A'),
                        'color': colors['database']
                    })
            except (ValueError, TypeError):
                pass

        # 模型预测数据
        for model in selected_models:
            pred_key = f'{model}_prediction'
            if result.get(pred_key) is not None:
                try:
                    pred_shift = float(result[pred_key])
                    if -200 <= pred_shift <= 200:
                        all_shifts.append(pred_shift)
                        plot_data.append({
                            'compound': compound_id,
                            'shift': pred_shift,
                            'type': f'{model.upper()}预测',
                            'model': model,
                            'smiles': result['smiles'],
                            'solvent': 'DL预测',
                            'color': colors.get(model, '#d62728')  # 默认红色
                        })
                except (ValueError, TypeError):
                    pass

    if not all_shifts:
        return None

    # 创建图形
    fig = go.Figure()

    # 设置X轴范围
    min_shift = min(all_shifts) - 10
    max_shift = max(all_shifts) + 10
    x_range = np.linspace(min_shift, max_shift, 2000)

    # 添加基线
    baseline_y = np.zeros_like(x_range)
    fig.add_trace(go.Scatter(
        x=x_range, y=baseline_y,
        mode='lines',
        line=dict(color='#333333', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 生成峰形并添加到图中
    compounds = list(set([d['compound'] for d in plot_data]))
    legend_added = set()

    for compound in compounds:
        compound_data = [d for d in plot_data if d['compound'] == compound]

        for j, data in enumerate(compound_data):
            # 生成洛伦兹峰
            peak_center = data['shift']
            peak_width = plot_config['peak_width']
            peak_height = 1.0 - j * 0.1  # 不同类型的峰使用不同高度

            gamma = peak_width / 2
            y_peak = peak_height / (1 + ((x_range - peak_center) / gamma) ** 2)

            # 添加峰形曲线
            fig.add_trace(go.Scatter(
                x=x_range, y=y_peak,
                mode='lines',
                fill='tonexty' if j == 0 else None,
                line=dict(color=data['color'], width=2.5),
                fillcolor=f"rgba({int(data['color'][1:3], 16)}, {int(data['color'][3:5], 16)}, {int(data['color'][5:7], 16)}, 0.2)",
                name=data['type'] if data['type'] not in legend_added else '',
                showlegend=data['type'] not in legend_added,
                hovertemplate=f"<b>{data['compound']}</b><br>" +
                              f"{data['type']}: <b>{peak_center:.2f} ppm</b><br>" +
                              f"SMILES: {data['smiles'][:40]}{'...' if len(data['smiles']) > 40 else ''}<br>" +
                              f"溶剂: {data['solvent']}<br>" +
                              "<extra></extra>"
            ))
            legend_added.add(data['type'])

            # 添加峰位标注
            if plot_config['show_annotations']:
                fig.add_annotation(
                    x=peak_center,
                    y=peak_height + 0.15,
                    text=f"<b>{peak_center:.1f}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=data['color'],
                    font=dict(size=11, color=data['color'], family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=data['color'],
                    borderwidth=2,
                    borderpad=3
                )

    # 更新布局 - 修复图例可见性问题
    fig.update_layout(
        title=dict(
            text="<b>³¹P NMR 化学位移谱图</b>",
            font=dict(size=24, color='#2c3e50', family="Arial Black"),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title="<b>化学位移 (ppm)</b>",
            titlefont=dict(size=16, color='#2c3e50', family="Arial"),
            tickfont=dict(size=14, color='#2c3e50'),
            autorange='reversed',  # NMR习惯：从右到左
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            showline=True,
            linewidth=3,
            linecolor='#2c3e50',
            mirror=True,
            range=[max_shift, min_shift],
            zeroline=False,
            tick0=0,
            dtick=20  # 每20ppm一个主刻度
        ),
        yaxis=dict(
            title="<b>相对强度</b>",
            titlefont=dict(size=16, color='#2c3e50', family="Arial"),
            tickfont=dict(size=14, color='#2c3e50'),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            showline=True,
            linewidth=3,
            linecolor='#2c3e50',
            mirror=True,
            range=[-0.15, 1.4],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#333333'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color='#2c3e50'),
        # 修复图例可见性问题
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="white",  # 完全不透明的白色背景
            bordercolor="black",  # 黑色边框更明显
            borderwidth=3,  # 加粗边框
            font=dict(
                size=16,  # 更大字体
                color="black",  # 明确的黑色字体
                family="Arial Black, Arial, sans-serif"  # 粗体字体
            ),
            itemsizing="constant",
            itemwidth=40,  # 增加图例项宽度
            itemclick="toggleothers",  # 改善交互性
            itemdoubleclick="toggle"
        ),
        width=1000,
        height=600,
        margin=dict(l=80, r=200, t=100, b=80)  # 增加右边距为图例留出空间
    )

    return fig


def create_comparison_visualization_force_visible(results: List[Dict], selected_models: List[str],
                                                  use_database: bool = False) -> go.Figure:
    """创建模型对比可视化图表 - 强制可见图例"""
    import plotly.express as px
    import pandas as pd
    import plotly.graph_objects as go

    comparison_data = []
    for i, result in enumerate(results):
        compound_id = f"化合物 {i + 1}"

        # 收集所有预测值
        values = {}
        if use_database and result.get('database') is not None:
            try:
                values['数据库实验值'] = float(result['database'])
            except (ValueError, TypeError):
                pass

        for model in selected_models:
            pred_key = f'{model}_prediction'
            if result.get(pred_key) is not None:
                try:
                    values[f'{model.upper()}预测'] = float(result[pred_key])
                except (ValueError, TypeError):
                    pass

        for method, value in values.items():
            comparison_data.append({
                'compound': compound_id,
                'method': method,
                'shift': value,
                'smiles': result['smiles']
            })

    if not comparison_data:
        return None

    df = pd.DataFrame(comparison_data)

    # 统一的颜色方案 - 与NMR谱图保持一致
    colors = {
        'ATTENTIVEFP预测': '#1f77b4',  # 明亮蓝色
        'GCN预测': '#ff7f0e',         # 明亮橙色
        '数据库实验值': '#2ca02c'      # 明亮绿色
    }

    fig = go.Figure()

    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        fig.add_trace(go.Bar(
            name=method,
            x=method_data['compound'],
            y=method_data['shift'],
            marker_color=colors.get(method, '#d62728'),  # 默认红色
            hovertemplate='<b style="color: #000000;">%{x}</b><br>' +
                          f'<span style="color: #000000; font-weight: bold;">{method}: %{{y:.2f}} ppm</span><br>' +
                          '<extra></extra>',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=14,
                font_color="black"
            )
        ))

    fig.update_layout(
        title=dict(
            text="<b>不同方法的化学位移预测对比</b>",
            font=dict(size=20, color='#000000'),
            x=0.5
        ),
        xaxis=dict(
            title="<b>化合物</b>",
            titlefont=dict(size=14, color='#000000'),
            tickangle=45,
            tickfont=dict(size=12, color='#000000')
        ),
        yaxis=dict(
            title="<b>化学位移 (ppm)</b>",
            titlefont=dict(size=14, color='#000000'),
            tickfont=dict(size=12, color='#000000')
        ),
        margin=dict(r=80, t=100),  # 右边距
        plot_bgcolor='white',
        paper_bgcolor='white',
        barmode='group',
        height=600,
        font=dict(family="Arial, sans-serif", color='#000000'),
        # 使用默认图例 - 放置在图表内部
        showlegend=True,
        legend=dict(
            x=0.7,  # 图例位置在图表内部
            y=0.95,
            bgcolor="white",  # 完全不透明的白色背景
            bordercolor="black",
            borderwidth=2,  # 加粗边框
            font=dict(
                size=16,  # 更大字体
                color="black",  # 明确黑色
                family="Arial Black, Arial, sans-serif"  # 使用粗体字体
            ),
            itemsizing="constant",  # 固定图例项大小
            itemwidth=30  # 增加图例项宽度
        ),
        # 添加网格线
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray'
    )

    return fig






# 主应用程序
def main():
    # 页面标题
    st.markdown("""
    <div class="main-header">
        <h1>🧪 化学位移预测平台</h1>
        <p>基于图神经网络的³¹P NMR化学位移智能预测系统</p>
    </div>
    """, unsafe_allow_html=True)

    # 加载数据库
    database = load_database()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 🎛️ 控制面板")

        # 数据库状态
        if database is not None:
            st.markdown(f"""
            <div class="success-alert">
                ✅ 数据库已加载<br>
                📊 包含 <strong>{len(database)}</strong> 条记录
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-alert">
                ⚠️ 数据库加载失败
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 预测方法选择
        st.markdown("### 🔬 预测方法")
        use_database = st.checkbox("🗃️ 数据库查找", value=True, help="优先从实验数据库查找")

        st.markdown("**机器学习模型：**")
        use_attentivefp = st.checkbox("🧠 AttentiveFP", value=True, help="注意力机制图神经网络")
        use_gcn = st.checkbox("🔗 GCN", value=False, help="图卷积神经网络")

        selected_models = []
        if use_attentivefp:
            selected_models.append('attentivefp')
        if use_gcn:
            selected_models.append('gcn')

    # 主标签页
    main_tabs = st.tabs(["🔍 预测分析", "🗃️ 数据库浏览"])

    # 预测分析标签页
    with main_tabs[0]:
        # 主内容区域
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📝 输入SMILES")
            example_smiles = "CC(=O)Oc1ccccc1C(=O)O"

            smiles_input = st.text_area(
                "请输入SMILES字符串（每行一个）",
                height=120,
                placeholder=f"示例：{example_smiles}\n可输入多个分子..."
            )

        with col2:
            st.markdown("### 🚀 操作")
            st.markdown("""
            <div class="info-alert">
                💡 <strong>使用提示：</strong><br>
                • 每行输入一个SMILES<br>
                • 优先查找实验数据<br>
                • 支持多模型对比<br>
            </div>
            """, unsafe_allow_html=True)

        # 预测按钮
        predict_button = st.button("🔍 开始预测", type="primary", use_container_width=True)

        # 预测逻辑
        if predict_button:
            if smiles_input and (selected_models or use_database):
                with st.spinner('🔄 正在进行预测分析...'):
                    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                    results = predict_with_multiple_models(smiles_list, selected_models, database if use_database else None)

                    # 存储结果到session state
                    st.session_state.prediction_results = results
                    st.session_state.selected_models_state = selected_models
                    st.session_state.use_database_state = use_database

                    # 统计信息
                    st.markdown("### 📊 预测统计")
                    db_count = sum(1 for r in results if r['source'] == 'database')
                    ml_count = sum(1 for r in results if r['source'] == 'ml_models')

                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{len(results)}</h3>
                            <p>总化合物数</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with stat_cols[1]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{db_count}</h3>
                            <p>数据库匹配</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with stat_cols[2]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{ml_count}</h3>
                            <p>模型预测</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with stat_cols[3]:
                        accuracy = f"{(db_count / len(results) * 100):.1f}%" if results else "0%"
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>{accuracy}</h3>
                            <p>数据库覆盖率</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 详细结果展示
                    st.markdown("### 🧪 详细预测结果")

                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="compound-title">化合物 #{i + 1}</div>
                                <p><strong>SMILES:</strong> <code>{result['smiles']}</code></p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 预测结果展示
                            result_cols = []

                            if use_database and result['database'] is not None:
                                result_cols.append(("数据库实验值", result['database'], result.get('solvent', 'N/A')))

                            for model in selected_models:
                                pred_key = f'{model}_prediction'
                                if result.get(pred_key) is not None:
                                    result_cols.append((f"{model.upper()}预测", result[pred_key], "DL模型"))

                            if result_cols:
                                cols = st.columns(len(result_cols))
                                for j, (col, (method, value, source)) in enumerate(zip(cols, result_cols)):
                                    with col:
                                        try:
                                            value_float = float(value)
                                            color = "#2D5016" if "数据库" in method else "#667eea"
                                            st.markdown(f"""
                                            <div style="text-align: center; padding: 1rem; 
                                                       background: linear-gradient(135deg, {color}20, {color}10); 
                                                       border-radius: 8px; border: 2px solid {color};">
                                                <div class="model-badge" style="background: {color};">{method}</div>
                                                <div class="prediction-value" style="color: {color};">{value_float:.2f} ppm</div>
                                                <small style="color: #666;">溶剂: {source}</small>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        except (ValueError, TypeError):
                                            st.error(f"预测错误: {value}")

                            # 显示额外信息
                            if result.get('additional_info') and any(result['additional_info'].values()):
                                with st.expander("📋 详细信息"):
                                    info = result['additional_info']
                                    info_cols = st.columns(3)
                                    with info_cols[0]:
                                        if info.get('MW'):
                                            st.write(f"**分子量:** {info['MW']}")
                                        if info.get('P_oxidation_state'):
                                            st.write(f"**P氧化态:** {info['P_oxidation_state']}")
                                    with info_cols[1]:
                                        if info.get('P_valence'):
                                            st.write(f"**P价态:** {info['P_valence']}")
                                        if info.get('envlab'):
                                            st.write(f"**环境标签:** {info['envlab']}")
                                    with info_cols[2]:
                                        if info.get('num'):
                                            st.write(f"**编号:** {info['num']}")

            else:
                st.warning("请输入SMILES并选择至少一种预测方法")

        # 可视化部分
        if hasattr(st.session_state, 'prediction_results'):
            st.markdown("---")
            st.markdown("### 📈 可视化分析")

            viz_tabs = st.tabs(["🌊 NMR谱图", "📋 数据表格"])

            with viz_tabs[0]:
                st.markdown("#### ³¹P NMR 化学位移谱图")

                # 使用优化的NMR谱图配置
                plot_config = {
                    'peak_width': 0.04,
                    'show_annotations': True,
                    'plot_style': "professional"
                }

                nmr_fig = create_enhanced_nmr_plot(
                    st.session_state.prediction_results,
                    st.session_state.selected_models_state,
                    st.session_state.use_database_state,
                    plot_config
                )

                if nmr_fig:
                    st.plotly_chart(nmr_fig, use_container_width=True)

                    # 谱图说明
                    st.markdown("""
                    <div class="info-alert">
                        <strong>谱图说明：</strong><br>
                        • X轴：化学位移 (ppm)，按NMR习惯从右到左排列<br>
                        • Y轴：相对强度，峰高代表信号强度<br>
                        • 不同颜色代表不同的预测方法<br>
                        • 鼠标悬停可查看详细信息
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("无法生成NMR谱图，请检查预测结果")

            with viz_tabs[1]:
                st.markdown("#### 预测结果数据表格")

                # 创建详细的数据表格
                table_data = []
                for i, result in enumerate(st.session_state.prediction_results):
                    row = {
                        '化合物ID': f'化合物 {i + 1}',
                        'SMILES': result['smiles'],
                        '数据来源': '数据库' if result['source'] == 'database' else 'DL模型'
                    }

                    if st.session_state.use_database_state and result.get('database') is not None:
                        row['数据库实验值 (ppm)'] = result['database']
                        row['溶剂'] = result.get('solvent', 'N/A')

                    for model in st.session_state.selected_models_state:
                        pred_key = f'{model}_prediction'
                        if result.get(pred_key) is not None:
                            try:
                                row[f'{model.upper()}预测 (ppm)'] = f"{float(result[pred_key]):.2f}"
                            except (ValueError, TypeError):
                                row[f'{model.upper()}预测 (ppm)'] = "错误"

                    table_data.append(row)

                if table_data:
                    df_display = pd.DataFrame(table_data)
                    st.dataframe(df_display, use_container_width=True)

                    # 导出功能
                    csv_data = df_display.to_csv(index=False)
                    st.download_button(
                        label="📁 下载CSV文件",
                        data=csv_data,
                        file_name="chemical_shift_predictions.csv",
                        mime="text/csv"
                    )

    # 数据库浏览标签页
    with main_tabs[1]:
        st.markdown("### 🗃️ 数据库浏览与检索")

        if database is not None:
            st.markdown(f"""
            <div class="info-alert">
                📊 <strong>数据库统计信息</strong><br>
                • 总记录数: {len(database)}<br>
                • 化学位移范围: {database['shift'].min():.1f} - {database['shift'].max():.1f} ppm<br>
                • 平均化学位移: {database['shift'].mean():.1f} ppm
            </div>
            """, unsafe_allow_html=True)

            # 数据库搜索功能
            st.markdown("##### 🔍 搜索与筛选")
            search_col1, search_col2 = st.columns(2)

            with search_col1:
                search_smiles = st.text_input("按SMILES搜索", placeholder="输入SMILES字符串")
            with search_col2:
                shift_range = st.slider("化学位移范围 (ppm)",
                                        float(database['shift'].min()),
                                        float(database['shift'].max()),
                                        (float(database['shift'].min()), float(database['shift'].max())))

            # 过滤数据库
            filtered_db = database.copy()

            if search_smiles:
                filtered_db = filtered_db[
                    filtered_db['cansmi'].str.contains(search_smiles, case=False, na=False)]

            filtered_db = filtered_db[(filtered_db['shift'] >= shift_range[0]) &
                                      (filtered_db['shift'] <= shift_range[1])]

            st.markdown(f"##### 📋 搜索结果 ({len(filtered_db)} 条记录)")

            if len(filtered_db) > 0:
                # 显示筛选后的数据
                display_columns = ['cansmi', 'shift', 'LM', 'MW', 'P_oxidation_state', 'P_valence', 'envlab']
                available_columns = [col for col in display_columns if col in filtered_db.columns]

                # 重命名列名为中文
                column_names = {
                    'cansmi': 'SMILES',
                    'shift': '化学位移 (ppm)',
                    'LM': '溶剂',
                    'MW': '分子量',
                    'P_oxidation_state': 'P氧化态',
                    'P_valence': 'P价态',
                    'envlab': '环境标签'
                }

                display_df = filtered_db[available_columns].rename(columns=column_names)

                # 分页显示
                page_size = 20
                total_pages = (len(display_df) - 1) // page_size + 1

                if total_pages > 1:
                    page_num = st.selectbox("选择页面", range(1, total_pages + 1)) - 1
                    start_idx = page_num * page_size
                    end_idx = start_idx + page_size
                    display_df = display_df.iloc[start_idx:end_idx]

                st.dataframe(display_df, use_container_width=True)

                # 数据库统计图表
                st.markdown("##### 📊 数据分布图")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # 化学位移分布直方图
                    import plotly.graph_objects as go
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=filtered_db['shift'],
                        nbinsx=30,
                        name='化学位移分布',
                        marker_color='#667eea',
                        opacity=0.8
                    ))
                    fig_hist.update_layout(
                        title='化学位移分布',
                        xaxis_title='化学位移 (ppm)',
                        yaxis_title='频次',
                        height=400,
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with chart_col2:
                    # 溶剂分布饼图
                    if 'LM' in filtered_db.columns:
                        solvent_counts = filtered_db['LM'].value_counts().head(10)
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=solvent_counts.index,
                            values=solvent_counts.values,
                            marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                        )])
                        fig_pie.update_layout(
                            title='溶剂分布 (前10种)',
                            height=400
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                # 导出筛选结果
                csv_filtered = display_df.to_csv(index=False)
                st.download_button(
                    label="📁 下载筛选结果",
                    data=csv_filtered,
                    file_name=f"filtered_database_{len(display_df)}_records.csv",
                    mime="text/csv"
                )

            else:
                st.warning("未找到匹配的记录，请调整搜索条件")
        else:
            st.error("数据库未加载，无法浏览数据")

    # 页面底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h4>🧪 关于本平台</h4>
        <p>本平台使用先进的图神经网络技术预测³¹P NMR化学位移，结合实验数据库为化学研究提供准确、快速的预测服务。</p>
        <div style="margin-top: 1rem;">
            <span class="model-badge" style="background: #2E86AB;">AttentiveFP模型</span>
            <span class="model-badge" style="background: #A23B72;">GCN模型</span>
            <span class="model-badge" style="background: #2D5016;">实验数据库</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()