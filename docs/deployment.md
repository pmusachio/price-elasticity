# Entrega e consumo

Este projeto entrega `reports/price_elasticity.csv` e um simulador Streamlit para explorar cenarios de desconto, aumento de preco e impacto de faturamento.

## Painel Streamlit

O app em `app/streamlit_app.py` possui:

- Aba de elasticidade com grafico Preco x Demanda e linha de tendencia.
- Painel de mudanca de preco com slider de aumento/desconto.
- Ranking de produtos por sensibilidade a preco.
- Aba opcional de elasticidade cruzada quando a matriz historica esta disponivel em `data/raw/df_crossprice_previous.csv`.

## Streamlit local

```bash
python -m pip install -r requirements.txt -r requirements-app.txt
PYTHONPATH=src python -m price_elasticity.cli analyze
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Notebook de consumo

O notebook `notebooks/05_deployment_and_consumption.ipynb` detalha os canais de entrega deste projeto:

- Streamlit para simulacao de elasticidade e impacto de desconto.
- Grafico Preco x Demanda, ranking e painel de mudanca de preco.
- CSV analitico com elasticidades por produto.
