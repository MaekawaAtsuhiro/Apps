import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#アプリ実行：ターミナルでstreamlit run whether_app.pyを入力
#過去の分の表示と予測結果を表示

kinki_whether_week=pd.read_csv('kinki_whether_week.csv')
kinki_whether_week["date"] = pd.to_datetime(kinki_whether_week["date"])

kinki_whether_month=pd.read_csv('kinki_whether_month.csv')
kinki_whether_month["date"] = pd.to_datetime(kinki_whether_month["date"])

def Weekly(whether):
    # 地名リストの取得
    areas = whether['地名'].unique()

    # 地名選択のウィジェット
    selected_area = st.selectbox('確認したい地名を選択してください', areas)

    # 選択された地名のデータを抽出
    selected_data = whether[whether['地名'] == selected_area]

    # 気象情報の選択のウィジェット
    selected_feature = st.selectbox('確認したい気象情報を選択してください', ['temperature(℃)', 'rainfall(mm)', 'daylight hours(h)', 'wind speed(m/s)','vapor pressure(hPa)'])

    # 時系列プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=selected_data["date"], y=selected_data[selected_feature], label=selected_feature, ax=ax,color='blue')
    ax.set_title(f'Weekly average {selected_feature} movement')
    ax.set_xlabel('date')
    ax.set_ylabel(selected_feature)
    ax.legend()  # 凡例を表示

    # プロットをStreamlitに表示
    st.pyplot(fig)

    # 任意の統計情報を表示
    st.write(f'{selected_area}の{selected_feature}の統計情報:')
    st.write(selected_data[selected_feature].describe())

    st.markdown("""
                ### 特定の期間が見れるようにします
                """)
    #特定の期間を見れるようにする
    st.write('(例)：2015/1/1のように入力してください')
    
    user_input_start = st.text_input("開始日付を入力してください", "")
    user_input_end= st.text_input("終了日付を入力してください", "")
    # 日時型に変換
    user_input_start = pd.to_datetime(user_input_start, errors='coerce')
    user_input_end = pd.to_datetime(user_input_end, errors='coerce')
    whether_se = selected_data[(selected_data['date'] >= user_input_start) & (selected_data['date'] <= user_input_end)]

    # 時系列プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=whether_se["date"], y=whether_se[selected_feature], label=selected_feature, ax=ax,color='blue')
    ax.set_title(f'Weekly average {selected_feature} movement')
    ax.set_xlabel('date')
    ax.set_ylabel(selected_feature)
    ax.legend()  # 凡例を表示

    # プロットをStreamlitに表示
    st.pyplot(fig)

    st.markdown("""
                ### 予測結果を表示します
                """)
    #特定の期間を見れるようにする
    st.write('青線が実データ、緑線が予測結果、灰色の範囲が信頼区間')

    # モデルのトレーニングとテストデータの分割
    selected_data['date']=pd.to_datetime(selected_data['date'])
    train, test = train_test_split(selected_data[selected_feature], test_size=0.2)
    future_size=len(test)

    # ARIMAモデルの構築
    model = pm.auto_arima(train, seasonal=True, suppress_warnings=True)

    # テストデータに対する予測
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

    # 予測結果の可視化
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(selected_data['date'][:-future_size], selected_data[selected_feature][:-future_size], label='Actual',color='blue')
    ax.plot(selected_data['date'][-future_size:], forecast, label='Forecast', linestyle='dashed',color='green')
    ax.fill_between(selected_data['date'][-future_size:], conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='Confidence Interval')
    ax.legend()

    # プロットをStreamlitに表示
    st.pyplot(fig)

def Monthly(whether):
    # 地名リストの取得
    areas = whether['地名'].unique()

    # 地名選択のウィジェット
    selected_area = st.selectbox('確認したい地名を選択してください', areas)

    # 選択された地名のデータを抽出
    selected_data = whether[whether['地名'] == selected_area]

    # 気象情報の選択のウィジェット
    selected_feature = st.selectbox('確認したい気象情報を選択してください', ['temperature(℃)', 'rainfall(mm)', 'daylight hours(h)', 'wind speed(m/s)','vapor pressure(hPa)'])

    # 時系列プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=selected_data["date"], y=selected_data[selected_feature], label=selected_feature, ax=ax,color='red')
    ax.set_title(f'Monthly average {selected_feature} movement')
    ax.set_xlabel('date')
    ax.set_ylabel(selected_feature)
    ax.legend()  # 凡例を表示

    # プロットをStreamlitに表示
    st.pyplot(fig)

    # 任意の統計情報を表示
    st.write(f'{selected_area}の{selected_feature}の統計情報:')
    st.write(selected_data[selected_feature].describe())

    st.markdown("""
                ### 特定の期間が見れるようにします
                """)
    #特定の期間を見れるようにする
    st.write('(例)：2015/1/1のように入力してください')
    
    user_input_start = st.text_input("開始日付を入力してください", "")
    user_input_end= st.text_input("終了日付を入力してください", "")
    # 日時型に変換
    user_input_start = pd.to_datetime(user_input_start, errors='coerce')
    user_input_end = pd.to_datetime(user_input_end, errors='coerce')
    whether_se = selected_data[(selected_data['date'] >= user_input_start) & (selected_data['date'] <= user_input_end)]

    # 時系列プロット
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=whether_se["date"], y=whether_se[selected_feature], label=selected_feature, ax=ax,color='red')
    ax.set_title(f'Monthly average {selected_feature} movement')
    ax.set_xlabel('date')
    ax.set_ylabel(selected_feature)
    ax.legend()  # 凡例を表示

    # プロットをStreamlitに表示
    st.pyplot(fig)

    st.markdown("""
                ### 予測結果を表示します
                """)
    #特定の期間を見れるようにする
    st.write('青線が実データ、緑線が予測結果、灰色の範囲が信頼区間')

    # モデルのトレーニングとテストデータの分割
    selected_data['date']=pd.to_datetime(selected_data['date'])
    train, test = train_test_split(selected_data[selected_feature], test_size=0.2)
    future_size=len(test)

    # ARIMAモデルの構築
    model = pm.auto_arima(train, seasonal=True, suppress_warnings=True)

    # テストデータに対する予測
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

    # 予測結果の可視化
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(selected_data['date'][:-future_size], selected_data[selected_feature][:-future_size], label='Actual',color='blue')
    ax.plot(selected_data['date'][-future_size:], forecast, label='Forecast', linestyle='dashed',color='green')
    ax.fill_between(selected_data['date'][-future_size:], conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='Confidence Interval')
    ax.legend()

    # プロットをStreamlitに表示
    st.pyplot(fig)

def view():
    st.title('地域ごとの気象情報⛅')
    st.write('temperature(℃)は気温　　rainfall(mm)は降水量')
    st.write('daylight hours(h)は日照時間　　wind speed(m/s)は風速')
    st.write('vapor pressure(hPa)は蒸気圧')

    st.markdown("""
                ## 2013年から2023年までの気象情報を表示します
                """)


    # 気象情報の選択のウィジェット
    selected_plot = st.selectbox('気象情報の表示方法を選択してください', ['週次データで表示したい','月次データで表示したい'])
    if selected_plot=='週次データで表示したい':
        Weekly(kinki_whether_week)
    else:
        Monthly(kinki_whether_month)
view()