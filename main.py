from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import prompt

# Carregar produtos comprados
product = pd.read_csv('products.csv')
sales = pd.read_csv('sales.csv')
product_sales = [product.split(',') for product in sales['produtos_comprados']]

te = TransactionEncoder()
te_ary = te.fit(product_sales).transform(product_sales)
df_transacoes = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_transacoes, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print(rules[['antecedents', 'consequents', 'confidence']])

product_list = product['produto'].tolist()
completer = WordCompleter(product_list)


def recommend(user_product, rules):
    recommend = set()

    for product in user_product:
        itens = rules[rules['antecedents'].apply(lambda x, product=product: product in x)]['consequents'].tolist()
        for item in itens:
            recommend.update(item)
    recommend -= set(user_product)
    return list(recommend)

while True:
    produtos_inseridos = prompt('Insira os produtos que você comprou (separados por vírgula): ', completer=completer)
    produtos_sugeridos = recommend(produtos_inseridos.split(','), rules)
    print(f'Produtos sugeridos: {produtos_sugeridos}')

# ----------------------- #
