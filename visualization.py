import plotly.graph_objects as go
import numpy as np
import streamlit as st


def plot_matrix_heatmap(matrix, title="Визуализация матрицы"):
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Строка: %{y}<br>Столбец: %{x}<br>Значение: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Столбцы",
        yaxis_title="Строки",
        width=500,
        height=500
    )

    return fig


def plot_eigenvalues(eigenvalues):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.real(eigenvalues),
        y=np.imag(eigenvalues),
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Собственные значения'
    ))

    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Единичная окружность'
    ))

    fig.update_layout(
        title="Собственные значения на комплексной плоскости",
        xaxis_title="Re(λ)",
        yaxis_title="Im(λ)",
        showlegend=True
    )

    return fig