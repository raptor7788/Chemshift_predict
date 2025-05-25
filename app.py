import torch
import streamlit as st
import numpy as np
from torch_geometric.utils import from_smiles
from torch_geometric.nn import AttentiveFP, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


# Load the models
def load_attentivefp_model():
    # 定义AttentiveFP模型（需与训练时参数一致）
    model = AttentiveFP(
        in_channels=9,  # 节点特征维度
        hidden_channels=66,  # 隐藏层维度
        out_channels=1,  # 输出维度
        edge_dim=3,  # 边特征维度
        num_layers=5,  # 层数
        num_timesteps=3,  # 时间步数
        dropout=0.0446259777448801  # dropout率
    )
    # 加载state_dict
    model.load_state_dict(
        torch.load("Resources/AttentiveFP_model.pt", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model


import torch
from torch_geometric.nn import GCN, global_mean_pool, global_max_pool, global_add_pool


class NMRShiftModel(torch.nn.Module):
    def __init__(
            self,
            in_channels=9,
            hidden_channels=185,  # Match your load_gcn_model
            num_layers=4,  # Match checkpoint (convs.0 to convs.3)
            out_channels=1,
            dropout=0.11080081715730111,  # Match your load_gcn_model
            pooling_method='max',  # Match your load_gcn_model
            act='relu',
            jk='cat'  # Changed to 'cat' to match gcn.lin.weight shape [185, 740]
    ):
        super().__init__()
        # Use PyG's GCN model
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,  # Output to hidden_channels first
            dropout=dropout,
            norm='batch_norm',
            jk=jk,
            act=act
        )
        # Final linear layer to get to out_channels
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.pooling_method = pooling_method

    def forward(self, x, edge_index, batch):
        # Apply GCN
        x = self.gcn(x, edge_index)

        # Apply global pooling based on the selected method
        if self.pooling_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_method == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling_method == 'max':
            x = global_max_pool(x, batch)

        # Apply final linear layer
        x = self.lin(x)
        return x


def load_gcn_model():
    # Define GCN model with parameters matching the checkpoint
    model = NMRShiftModel(
        in_channels=9,
        hidden_channels=185,
        num_layers=4,
        out_channels=1,
        dropout=0.11080081715730111,
        pooling_method='max',
        act='relu',
        jk='cat'  # Changed to 'cat'
    )
    # Load state_dict
    model.load_state_dict(
        torch.load("Resources/GCN_model.pt", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model

# Function to inverse transform the scaled predictions to original scale
def inverse_transform(scaled_value):
    # 使用提供的中心值和缩放因子进行逆变换
    center = 23.47
    scale = 22.69999886
    return scaled_value * scale + centerss


# Function to predict chemical shift
def predict_chemical_shift(smiles_list, model_type='attentivefp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the selected model
    if model_type == 'attentivefp':
        model = load_attentivefp_model().to(device)
    else:  # gcn
        model = load_gcn_model().to(device)

    predictions = []
    raw_predictions = []
    for smile in smiles_list:
        try:
            # Convert SMILES to graph
            g = from_smiles(smile)
            g.x = g.x.float()
            g = g.to(device)

            # Add batch information
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)

            # Perform prediction based on model type
            with torch.no_grad():
                if model_type == 'attentivefp':
                    pred = model(g.x, g.edge_index, g.edge_attr, g.batch)
                else:  # gcn
                    pred = model(g.x, g.edge_index, g.batch)

            # Get raw prediction value
            raw_pred = pred.item()
            raw_predictions.append(raw_pred)

            # Apply inverse transform to get the actual chemical shift value
            actual_pred = inverse_transform(raw_pred)
            predictions.append(actual_pred)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
            raw_predictions.append(None)

    return predictions, raw_predictions


# Page configuration
st.set_page_config(
    page_title="Chemical Shift Prediction",
    page_icon="🧪",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("导航菜单")
page = st.sidebar.radio("选择页面", ["首页", "关于"])

if page == "首页":
    # Streamlit UI for prediction
    st.title('化学位移预测 (Chemical Shift Prediction)')
    st.write('请在下方输入SMILES字符串，点击"预测"按钮获取化学位移值。')

    # Example SMILES
    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    st.write(f"示例SMILES: `{example_smiles}`")

    smiles_input = st.text_area("SMILES字符串 (每行一个)", height=150)

    # Model selection
    model_type = st.radio(
        "选择预测模型",
        ["AttentiveFP", "GCN"],
        horizontal=True,
        help="AttentiveFP: 使用注意力机制的图神经网络; GCN: 图卷积神经网络"
    )

    # Convert model selection to lowercase for internal use
    model_type_lower = model_type.lower()

    # Option to show normalized values
    show_normalized = st.checkbox("同时显示归一化值", value=False)

    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button("预测", type="primary")

    if predict_button:
        if smiles_input:
            with st.spinner('正在进行预测...'):
                smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                shift_predictions, raw_predictions = predict_chemical_shift(smiles_list, model_type_lower)

                st.write(f"### 预测结果 (使用 {model_type} 模型)")

                # Display results without using st.table
                for i, (smile, pred, raw_pred) in enumerate(zip(smiles_list, shift_predictions, raw_predictions)):
                    result_container = st.container()
                    with result_container:
                        if show_normalized and not isinstance(pred, str):
                            cols = st.columns([1, 3, 2, 2])
                            with cols[0]:
                                st.write(f"**#{i + 1}**")
                            with cols[1]:
                                st.write(f"**SMILES:** `{smile}`")
                            with cols[2]:
                                if isinstance(pred, float):
                                    st.write(f"**预测的化学位移:** {pred:.4f} ppm")
                                else:
                                    st.write(f"**错误:** {pred}")
                            with cols[3]:
                                if raw_pred is not None:
                                    st.write(f"**归一化值:** {raw_pred:.4f}")
                        else:
                            cols = st.columns([1, 4, 2])
                            with cols[0]:
                                st.write(f"**#{i + 1}**")
                            with cols[1]:
                                st.write(f"**SMILES:** `{smile}`")
                            with cols[2]:
                                if isinstance(pred, float):
                                    st.write(f"**预测的化学位移:** {pred:.4f} ppm")
                                else:
                                    st.write(f"**错误:** {pred}")
                    st.divider()
        else:
            st.warning("请至少输入一个SMILES字符串。")

elif page == "关于":
    # About page content
    st.title("关于本应用")
    st.write("""
    本Web应用程序使用图神经网络(GNN)模型从分子的SMILES表示预测化学位移。

    ### 工作原理
    1. **SMILES输入**: 用户可以在文本区域中输入一个或多个SMILES字符串(每行一个)。
    2. **选择模型**: 用户可以选择使用AttentiveFP或GCN模型进行预测。
    3. **预测过程**: 应用程序将这些SMILES字符串转换为图表示，并将它们输入预训练的模型。
    4. **输出结果**: 模型为给定的SMILES字符串预测化学位移值。
    5. **数据归一化**: 模型输出的预测值会通过逆归一化转换为实际的化学位移值(ppm)。使用的归一化参数:
       - 中心值(center): 23.47
       - 缩放因子(scale): 22.69999886

    ### 关于化学位移
    化学位移是核磁共振(NMR)谱学中的一个重要参数，它表示分子中特定原子核的共振频率与参考物质的偏差。化学位移受分子结构影响，可用于分子结构鉴定和分析。

    ### 关于模型
    本应用提供两种不同的图神经网络模型:

    #### AttentiveFP模型参数:
    - 节点特征维度: 9
    - 隐藏层维度: 66
    - 输出维度: 1
    - 边特征维度: 3
    - 层数: 5
    - 时间步数: 3
    - Dropout率: 0.0446259777448801

    #### GCN模型参数:
    - 节点特征维度: 9
    - 隐藏层维度: 185
    - 输出维度: 1
    - 层数: 4
    - Dropout率: 0.11080081715730111
    - 池化方法: max
    - 激活函数: relu
    """)

    st.write("### 技术栈")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**PyTorch & PyTorch Geometric**\n\n用于图神经网络的训练和推理")
    with col2:
        st.info("**Streamlit**\n\n用于构建交互式Web界面")
    with col3:
        st.info("**RDKit**\n\n用于分子处理和SMILES转换")

