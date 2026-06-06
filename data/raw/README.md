# Dados

Fonte Kaggle: [Ecommerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).

Arquivos esperados nesta pasta:

- `df_ready.csv`

- O projeto espera uma base analitica em nivel produto-data com preco, preco com desconto e demanda observada.
- Se voce partir dos eventos brutos do Kaggle, gere `df_ready.csv` com as colunas descritas em `data/raw/README.md` antes de rodar a analise.

## Preparacao

Crie ou envie `df_ready.csv` para esta pasta com as colunas `name`, `disc_price` e `imp_count`, alem das demais colunas descritivas usadas na analise.

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
