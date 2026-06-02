import plotly.graph_objects as go


def build_line_chart(history, forecast, lower, upper, title='Harga Historis dan Prediksi'):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history['Tanggal'],
            y=history['target'],
            name='Harga Historis',
            mode='lines',
            line=dict(color='#7fd1b9', width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast['date'],
            y=forecast['mean'],
            name='Prediksi',
            mode='lines',
            line=dict(color='#f8d57e', width=3, dash='dot'),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast['date'],
            y=upper,
            name='Confidence Upper',
            mode='lines',
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast['date'],
            y=lower,
            name='Confidence Lower',
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(248, 213, 126, 0.18)',
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='Tanggal',
        yaxis_title='Harga (Rp/kg)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eef5ff'),
        legend=dict(bgcolor='rgba(255,255,255,0.04)', bordercolor='rgba(255,255,255,0.12)'),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
    return fig
