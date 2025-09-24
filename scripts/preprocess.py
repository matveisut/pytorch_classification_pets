import json
import os
from sklearn.model_selection import train_test_split
 

def split_by_class(df, class_column, test_size=0.2, random_state=42):
    """
    Разделяет данные по классам на train и test
    
    Parameters:
    df - исходный DataFrame
    class_column - название столбца с классами
    test_size - доля тестовой выборки (по умолчанию 0.2)
    random_state - для воспроизводимости результатов
    """
    
    # Создаем новые столбцы
    df['split'] = ''
    
    # Для каждого уникального класса
    for class_name in df[class_column].unique():
        # Выбираем данные только этого класса
        class_data = df[df[class_column] == class_name]
        
        # Разделяем на train и test
        train_idx, test_idx = train_test_split(
            class_data.index, 
            test_size=test_size, 
            random_state=random_state,
            shuffle= True
        )
        
        # Помечаем разделение
        df.loc[train_idx, 'split'] = 'train'
        df.loc[test_idx, 'split'] = 'test'
    
    return df