import ta
import pandas as pd

class AddIndicators:
    
    def add_indicator(self, data, indicator_name, **kwargs):
        """
        Adiciona indicadores técnicos ao dataframe.
        
        Args:
            data (pd.DataFrame): O dataframe com os dados de mercado (deve conter colunas 'close', 'high', 'low', etc).
            indicator_name (str): Nome do indicador técnico a ser adicionado (ex.: 'rsi', 'macd', 'sma', 'bollinger').
            **kwargs: Parâmetros necessários para inicializar o indicador técnico.
        
        Returns:
            pd.DataFrame: Dataframe atualizado com as colunas do indicador técnico adicionado.
        """

        indicator_mapping = {
            "rsi": ta.momentum.RSIIndicator,
            "macd": ta.trend.MACD,
            "sma": ta.trend.SMAIndicator,
            "bollinger": ta.volatility.BollingerBands
        }

        if indicator_name not in indicator_mapping:
            raise ValueError(
                f"Indicador '{indicator_name}' não é suportado. "
                f"Indicadores disponíveis: {list(indicator_mapping.keys())}"
            )

        indicator = indicator_mapping[indicator_name](**kwargs)

        if indicator_name == "macd":
            data["macd"] = indicator.macd()
            data["macd_signal"] = indicator.macd_signal()
            data["macd_diff"] = indicator.macd_diff()

        elif indicator_name == "rsi":
            data["rsi"] = indicator.rsi()

        elif indicator_name == "sma":
            data["sma"] = indicator.sma_indicator()

        elif indicator_name == "bollinger":
            data["bollinger_mavg"] = indicator.bollinger_mavg()
            data["bollinger_hband"] = indicator.bollinger_hband()
            data["bollinger_lband"] = indicator.bollinger_lband()
            data["bollinger_width"] = indicator.bollinger_wband()

        return data
        