import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
import uvicorn

N = 5  # Количество рекомендаций для пользователя
PATH = r'C:\Users\blinov.ka\Desktop\пары\рекоменд\data' # Путь до папки data

router = APIRouter()

def get_df():
    '''
    - Читает данные из Excel файлов, представляет их в виде Pandas DataFrame.
    - Производит предобработку данных, преобразуя их в матрицы, вычисляет косинусное расстояние между товарами.
    '''
    # Чтение данных из файлов Excel и преобразование их в DataFrames
    df_2 = pd.read_excel(os.path.join(PATH, '3. Обработанные данные (используются в функциях)/товары.xlsx'))
    df_3 = pd.read_excel(os.path.join(PATH, '3. Обработанные данные (используются в функциях)/только_оценки.xlsx'))
    # Создание параметров для расчета рейтингов и матрицы признаков
    parametrs = pd.get_dummies(df_2, columns = ['brand','material_category','gender_category', 'rank_category'])
    ratings = parametrs[['asin','СРЕДНЯЯ ОЦЕНКА']]
    parametrs = parametrs.drop(['asin','СРЕДНЯЯ ОЦЕНКА'], axis=1)
    matrix = parametrs.to_numpy()
    avg_ratings = df_3.groupby(['пользователи', 'товар'], as_index=False).mean()
    top_rated_products = avg_ratings.sort_values(by='оценка', ascending=False).groupby('пользователи').head(1)
    # Вычисление косинусного расстояния между товарами на основе их признаков
    rasstoyanie = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            dot_product = np.dot(matrix[i], matrix[j])
            norm_product = np.linalg.norm(matrix[i]) * np.linalg.norm(matrix[j])
            sim = dot_product / norm_product
            cos_dis = 1 - sim
            if cos_dis == 1:
                rasstoyanie[i][j] = 0
            else:
                rasstoyanie[i][j] = cos_dis

        return top_rated_products, ratings, rasstoyanie

def generate_recommendations_for_user(user_id):
    '''
    - Получает топ-рейтинговые товары, рейтинги и матрицу расстояний.
    - Проверяет наличие пользователя в топе рейтинговых товаров.
    - Генерирует рекомендации для пользователя на основе похожих товаров.
    '''

    top_rated_products,ratings,rasstoyanie = get_df()

    if user_id not in top_rated_products['пользователи'].unique():
        return ["Пользователь не найден или для него нет рекомендаций"]

    user_top_product = top_rated_products[top_rated_products['пользователи'] == user_id]['товар'].iloc[0]
    product_index = ratings[ratings['asin'] == user_top_product].index[0]
    similar_products_indices = np.argsort(rasstoyanie[product_index])[1:N+1]
    similar_products_asin = ratings.iloc[similar_products_indices]['asin']
    recommended_products = ratings[ratings['asin'].isin(similar_products_asin)].sort_values(by='СРЕДНЯЯ ОЦЕНКА', ascending=False).head(N)
    return recommended_products['asin'].tolist(), recommended_products['СРЕДНЯЯ ОЦЕНКА'].tolist()  # Возвращаем список ASIN рекомендованных товаров

def recommend(name,item_name):
    '''
    - Функция рассчитывает ожидаемую оценку пользователем "name" товара "item_name"
    '''
    # Создание параметров для расчета рейтингов и матрицы признаков
    df_1 = pd.read_excel(os.path.join(PATH, '3. Обработанные данные (используются в функциях)/матрица_оценок.xlsx'))
    list_ids = list_ids = df_1['Unnamed: 0'].tolist()
    items = df_1.columns.values.tolist()
    df_1 = df_1.fillna(0)
    df_1= df_1.drop(['Unnamed: 0'], axis=1)
    matrix = df_1.to_numpy()
    matrix = matrix.T
    rasstoyanie = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            dis_ = np.linalg.norm(matrix[i] - matrix[j])
            # квадрат евклидова расстояния
            cos_dis = dis_ * dis_ 
            if cos_dis == 1:
                rasstoyanie[i][j] = 0
            else:
                rasstoyanie[i][j] = cos_dis
    # Определяем человека
    # Вводим его индификатор
    person = list_ids.index(name)
    f = 1
    # Если у него нет оценок
    for i in matrix[person - 1]:
        if i != 0:
            f = 0
    if f:
        return 'Покупатель с нулевыми оценками'
    # Средняя оценка этого пользователя
    not_zero = len([item for item in matrix[person-1] if item !=0])
    r_a = sum(matrix[person-1])/not_zero
    # Товар который необходимо оценить
    
    item = items.index(item_name)
    
    # Порог схожести
    porog = float(0.8)
    summa_dis = 0
    # Cохраним средние оценки и расстояния
    avg_r = []
    avg_u_i = 0
    cnt = 0
    for i,j in enumerate(rasstoyanie[person-1]):
        if j > porog:
    # Также пользователь должен оценить товар
            if matrix[i][item-1] != 0:
                cnt += 1
                avg_u_i += matrix[i][item-1]
    # Накапливаем сумму расстояний для знаменателя
                summa_dis += j
    # Средняя оценка данного пользователя (без учета нулей)
                no_zero = len([item for item in matrix[i] if item !=0])
                r_u = sum(matrix[i])/no_zero
                avg_r.append((r_u, j))
    try:
        r_u_i =  avg_u_i/cnt
    except ZeroDivisionError:
         return 'Близких пользователей с оценкой товара нет. Попробуйте понизить порог схожести'
    # Высчитываем среднюю оценку по формуле
    summ = 0
    for i in avg_r:
        a = (r_u_i - i[0]) * i[1]
        summ += a
    try:
        final = round(r_a + (summ/summa_dis))
    except ZeroDivisionError:
        final = -1
    if final == -1:
        return 'Ожидаемую оценку предсказать невозможно'
    elif final >= 4:
        return f'Ожидаемая оценка клиентом {name} товара {item_name} равняется {final}. Рекомендация: предложить данный товар пользователю'
    else:
       return f'Ожидаемая оценка клиентом {name} товара {item_name} равняется {final}. Рекомендация: не предлагать данный товар пользователю'

@router.get("/recommendations/{user_id}")
async def read_recommendations(user_id: str):
    '''
    - Получить рекомендации для конкретного пользователя.
    '''
    recommendations, estimation = generate_recommendations_for_user(user_id)
    return {"user_id": user_id, "Рекомендуемые товары": recommendations, "Средняя оценка:": estimation}

@router.get("/")
async def mainpage() -> str:
    '''
    - Возвращает информацию о доступных маршрутах.
    '''
    return "Ты на главной странице получения рекомендаций. 1.Чтобы просмотреть всех пользователей: /users; 2.Получить рекомендацию: /recommendations/user_id;"

@router.get("/users")
async def get_users():
    '''
    - Возвращает список всех пользователей.
    '''
    df_1 = pd.read_excel(os.path.join(PATH, '3. Обработанные данные (используются в функциях)/матрица_оценок.xlsx'))
    list_ids = df_1['Unnamed: 0'].tolist()
    return list_ids

@router.get("/items")
async def get_items():
    '''
    - Возвращает список всех товаров.
    '''
    df_1 = pd.read_excel(os.path.join(PATH, '3. Обработанные данные (используются в функциях)/матрица_оценок.xlsx'))
    items = df_1.columns.values.tolist()
    return items

@router.get("/recommendations/{user_id}/{item_id}")
async def read_recommendations_item(user_id: str, item_id: str):
    '''
    - Получить рекомендации для конкретного пользователя.
    '''
    result = recommend(user_id, item_id)
    return result

app = FastAPI()

# Включаем маршруты из router в основное приложение
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)