{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "i1h7LAlVa7Gu"
      },
      "source": [
        "Importing the Libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "d2o7jdWHXE6K"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn import metrics"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y1jC584Mbd4Q"
      },
      "source": [
        "Data Collection and Processing"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BQtjCTzHbZQO"
      },
      "source": [
        "# loading the csv data to a Pandas DataFrame\n",
        "gold_data = pd.read_csv('/content/gold price dataset.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "S5xeeB9LbyA9",
        "outputId": "a80fe09f-64e8-449d-b8d2-01a2b8919b82"
      },
      "source": [
        "# print first 5 rows in the dataframe\n",
        "gold_data.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Date</th>\n",
              "      <th>SPX</th>\n",
              "      <th>GLD</th>\n",
              "      <th>USO</th>\n",
              "      <th>SLV</th>\n",
              "      <th>EUR/USD</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1/2/2008</td>\n",
              "      <td>1447.160034</td>\n",
              "      <td>84.860001</td>\n",
              "      <td>78.470001</td>\n",
              "      <td>15.180</td>\n",
              "      <td>1.471692</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1/3/2008</td>\n",
              "      <td>1447.160034</td>\n",
              "      <td>85.570000</td>\n",
              "      <td>78.370003</td>\n",
              "      <td>15.285</td>\n",
              "      <td>1.474491</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1/4/2008</td>\n",
              "      <td>1411.630005</td>\n",
              "      <td>85.129997</td>\n",
              "      <td>77.309998</td>\n",
              "      <td>15.167</td>\n",
              "      <td>1.475492</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1/7/2008</td>\n",
              "      <td>1416.180054</td>\n",
              "      <td>84.769997</td>\n",
              "      <td>75.500000</td>\n",
              "      <td>15.053</td>\n",
              "      <td>1.468299</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1/8/2008</td>\n",
              "      <td>1390.189941</td>\n",
              "      <td>86.779999</td>\n",
              "      <td>76.059998</td>\n",
              "      <td>15.590</td>\n",
              "      <td>1.557099</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "       Date          SPX        GLD        USO     SLV   EUR/USD\n",
              "0  1/2/2008  1447.160034  84.860001  78.470001  15.180  1.471692\n",
              "1  1/3/2008  1447.160034  85.570000  78.370003  15.285  1.474491\n",
              "2  1/4/2008  1411.630005  85.129997  77.309998  15.167  1.475492\n",
              "3  1/7/2008  1416.180054  84.769997  75.500000  15.053  1.468299\n",
              "4  1/8/2008  1390.189941  86.779999  76.059998  15.590  1.557099"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "NrywfHOBb6HD",
        "outputId": "8167fc52-3dc2-4227-ba75-172e7e2c3b12"
      },
      "source": [
        "# print last 5 rows of the dataframe\n",
        "gold_data.tail()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Date</th>\n",
              "      <th>SPX</th>\n",
              "      <th>GLD</th>\n",
              "      <th>USO</th>\n",
              "      <th>SLV</th>\n",
              "      <th>EUR/USD</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2285</th>\n",
              "      <td>5/8/2018</td>\n",
              "      <td>2671.919922</td>\n",
              "      <td>124.589996</td>\n",
              "      <td>14.0600</td>\n",
              "      <td>15.5100</td>\n",
              "      <td>1.186789</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2286</th>\n",
              "      <td>5/9/2018</td>\n",
              "      <td>2697.790039</td>\n",
              "      <td>124.330002</td>\n",
              "      <td>14.3700</td>\n",
              "      <td>15.5300</td>\n",
              "      <td>1.184722</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2287</th>\n",
              "      <td>5/10/2018</td>\n",
              "      <td>2723.070068</td>\n",
              "      <td>125.180000</td>\n",
              "      <td>14.4100</td>\n",
              "      <td>15.7400</td>\n",
              "      <td>1.191753</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2288</th>\n",
              "      <td>5/14/2018</td>\n",
              "      <td>2730.129883</td>\n",
              "      <td>124.489998</td>\n",
              "      <td>14.3800</td>\n",
              "      <td>15.5600</td>\n",
              "      <td>1.193118</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2289</th>\n",
              "      <td>5/16/2018</td>\n",
              "      <td>2725.780029</td>\n",
              "      <td>122.543800</td>\n",
              "      <td>14.4058</td>\n",
              "      <td>15.4542</td>\n",
              "      <td>1.182033</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "           Date          SPX         GLD      USO      SLV   EUR/USD\n",
              "2285   5/8/2018  2671.919922  124.589996  14.0600  15.5100  1.186789\n",
              "2286   5/9/2018  2697.790039  124.330002  14.3700  15.5300  1.184722\n",
              "2287  5/10/2018  2723.070068  125.180000  14.4100  15.7400  1.191753\n",
              "2288  5/14/2018  2730.129883  124.489998  14.3800  15.5600  1.193118\n",
              "2289  5/16/2018  2725.780029  122.543800  14.4058  15.4542  1.182033"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vgnDjvpocdUp",
        "outputId": "13fb521f-29a7-401a-fac6-c3a12ff3668b"
      },
      "source": [
        "# number of rows and columns\n",
        "gold_data.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2290, 6)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9SEXC4AWcnDu",
        "outputId": "6c660677-2c14-4caa-afef-085e130f3e2d"
      },
      "source": [
        "# getting some basic informations about the data\n",
        "gold_data.info()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 2290 entries, 0 to 2289\n",
            "Data columns (total 6 columns):\n",
            " #   Column   Non-Null Count  Dtype  \n",
            "---  ------   --------------  -----  \n",
            " 0   Date     2290 non-null   object \n",
            " 1   SPX      2290 non-null   float64\n",
            " 2   GLD      2290 non-null   float64\n",
            " 3   USO      2290 non-null   float64\n",
            " 4   SLV      2290 non-null   float64\n",
            " 5   EUR/USD  2290 non-null   float64\n",
            "dtypes: float64(5), object(1)\n",
            "memory usage: 107.5+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tjmFVXi2cv4Q",
        "outputId": "2fa51b37-0af3-4ce0-963a-f48fba8e0a84"
      },
      "source": [
        "# checking the number of missing values\n",
        "gold_data.isnull().sum()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Date       0\n",
              "SPX        0\n",
              "GLD        0\n",
              "USO        0\n",
              "SLV        0\n",
              "EUR/USD    0\n",
              "dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 288
        },
        "id": "9IcOnRfhc7zv",
        "outputId": "fc921d3f-e836-4042-ed2c-e77ad4216e47"
      },
      "source": [
        "# getting the statistical measures of the data\n",
        "gold_data.describe()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>SPX</th>\n",
              "      <th>GLD</th>\n",
              "      <th>USO</th>\n",
              "      <th>SLV</th>\n",
              "      <th>EUR/USD</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>2290.000000</td>\n",
              "      <td>2290.000000</td>\n",
              "      <td>2290.000000</td>\n",
              "      <td>2290.000000</td>\n",
              "      <td>2290.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>1654.315776</td>\n",
              "      <td>122.732875</td>\n",
              "      <td>31.842221</td>\n",
              "      <td>20.084997</td>\n",
              "      <td>1.283653</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>519.111540</td>\n",
              "      <td>23.283346</td>\n",
              "      <td>19.523517</td>\n",
              "      <td>7.092566</td>\n",
              "      <td>0.131547</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>676.530029</td>\n",
              "      <td>70.000000</td>\n",
              "      <td>7.960000</td>\n",
              "      <td>8.850000</td>\n",
              "      <td>1.039047</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1239.874969</td>\n",
              "      <td>109.725000</td>\n",
              "      <td>14.380000</td>\n",
              "      <td>15.570000</td>\n",
              "      <td>1.171313</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>1551.434998</td>\n",
              "      <td>120.580002</td>\n",
              "      <td>33.869999</td>\n",
              "      <td>17.268500</td>\n",
              "      <td>1.303296</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>2073.010070</td>\n",
              "      <td>132.840004</td>\n",
              "      <td>37.827501</td>\n",
              "      <td>22.882499</td>\n",
              "      <td>1.369971</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>2872.870117</td>\n",
              "      <td>184.589996</td>\n",
              "      <td>117.480003</td>\n",
              "      <td>47.259998</td>\n",
              "      <td>1.598798</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "               SPX          GLD          USO          SLV      EUR/USD\n",
              "count  2290.000000  2290.000000  2290.000000  2290.000000  2290.000000\n",
              "mean   1654.315776   122.732875    31.842221    20.084997     1.283653\n",
              "std     519.111540    23.283346    19.523517     7.092566     0.131547\n",
              "min     676.530029    70.000000     7.960000     8.850000     1.039047\n",
              "25%    1239.874969   109.725000    14.380000    15.570000     1.171313\n",
              "50%    1551.434998   120.580002    33.869999    17.268500     1.303296\n",
              "75%    2073.010070   132.840004    37.827501    22.882499     1.369971\n",
              "max    2872.870117   184.589996   117.480003    47.259998     1.598798"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "f9SUQ8hodW4b"
      },
      "source": [
        "Correlation:\n",
        "1. Positive Correlation\n",
        "2. Negative Correlation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "C3xgji81dJUW"
      },
      "source": [
        "correlation = gold_data.corr()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 488
        },
        "id": "oOqb9j0Ad-Zx",
        "outputId": "629a76ce-d91f-459d-c07e-3ab88af34fcf"
      },
      "source": [
        "# constructing a heatmap to understand the correlatiom\n",
        "plt.figure(figsize = (8,8))\n",
        "sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7ff32443b350>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 576x576 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qtQ_icjRemI4",
        "outputId": "cec00681-c356-42e4-e4c6-557e6c2b8ccd"
      },
      "source": [
        "# correlation values of GLD\n",
        "print(correlation['GLD'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "SPX        0.049345\n",
            "GLD        1.000000\n",
            "USO       -0.186360\n",
            "SLV        0.866632\n",
            "EUR/USD   -0.024375\n",
            "Name: GLD, dtype: float64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 357
        },
        "id": "TMr-xVEwfIKg",
        "outputId": "20e55a70-f442-4821-d6a2-3c3bc9e01226"
      },
      "source": [
        "# checking the distribution of the GLD Price\n",
        "sns.distplot(gold_data['GLD'],color='green')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
            "  warnings.warn(msg, FutureWarning)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7ff316c5da90>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4bdwLbPEfqWI"
      },
      "source": [
        "Splitting the Features and Target"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SJNxCR0vfWxe"
      },
      "source": [
        "X = gold_data.drop(['Date','GLD'],axis=1)\n",
        "Y = gold_data['GLD']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qW9UvLSNf8zH",
        "outputId": "c61137e6-7ab4-491f-c626-d4bda00c48ce"
      },
      "source": [
        "print(X)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "              SPX        USO      SLV   EUR/USD\n",
            "0     1447.160034  78.470001  15.1800  1.471692\n",
            "1     1447.160034  78.370003  15.2850  1.474491\n",
            "2     1411.630005  77.309998  15.1670  1.475492\n",
            "3     1416.180054  75.500000  15.0530  1.468299\n",
            "4     1390.189941  76.059998  15.5900  1.557099\n",
            "...           ...        ...      ...       ...\n",
            "2285  2671.919922  14.060000  15.5100  1.186789\n",
            "2286  2697.790039  14.370000  15.5300  1.184722\n",
            "2287  2723.070068  14.410000  15.7400  1.191753\n",
            "2288  2730.129883  14.380000  15.5600  1.193118\n",
            "2289  2725.780029  14.405800  15.4542  1.182033\n",
            "\n",
            "[2290 rows x 4 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lKUe3C-qf9y8",
        "outputId": "cfcf051d-0ec2-4cd5-c504-ceb22b63b5c9"
      },
      "source": [
        "print(Y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0        84.860001\n",
            "1        85.570000\n",
            "2        85.129997\n",
            "3        84.769997\n",
            "4        86.779999\n",
            "           ...    \n",
            "2285    124.589996\n",
            "2286    124.330002\n",
            "2287    125.180000\n",
            "2288    124.489998\n",
            "2289    122.543800\n",
            "Name: GLD, Length: 2290, dtype: float64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nv8UohBVgE1Z"
      },
      "source": [
        "Splitting into Training data and Test Data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KkrUByFugBUn"
      },
      "source": [
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1vrCHktWgqfi"
      },
      "source": [
        "Model Training:\n",
        "Random Forest Regressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "N17qRKKGgoaZ"
      },
      "source": [
        "regressor = RandomForestRegressor(n_estimators=100)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DP2he4-PhMso",
        "outputId": "62c30f51-7e7c-425e-b7f2-47e38bf38513"
      },
      "source": [
        "# training the model\n",
        "regressor.fit(X_train,Y_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',\n",
              "                      max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
              "                      max_samples=None, min_impurity_decrease=0.0,\n",
              "                      min_impurity_split=None, min_samples_leaf=1,\n",
              "                      min_samples_split=2, min_weight_fraction_leaf=0.0,\n",
              "                      n_estimators=100, n_jobs=None, oob_score=False,\n",
              "                      random_state=None, verbose=0, warm_start=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SHNFVsr4hbG2"
      },
      "source": [
        "Model Evaluation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uOLpKKD_hXSl"
      },
      "source": [
        "# prediction on Test Data\n",
        "test_data_prediction = regressor.predict(X_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WSIqrLNdhnOr",
        "outputId": "72344c44-af1a-491f-bfc9-8046b55efee2"
      },
      "source": [
        "print(test_data_prediction)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[168.32699968  81.94819986 115.70480041 127.41010064 120.90700068\n",
            " 154.71489803 150.13359876 126.05480078 117.49259876 126.10170025\n",
            " 116.68400094 171.42870056 141.30849872 168.02159836 115.23710006\n",
            " 117.65880054 139.99590286 170.14650043 159.36550329 157.7788999\n",
            " 155.08109978 125.5072004  175.80919981 157.22360324 125.23650052\n",
            "  93.63189949  77.31300031 120.41329992 119.09699944 167.33849992\n",
            "  87.85020039 125.33850039  91.04200093 117.60520038 121.21729884\n",
            " 135.91539997 115.60030137 114.8546008  148.90609944 107.45720142\n",
            " 104.09990231  87.17429793 126.55180059 118.02069986 153.00689898\n",
            " 119.57450031 108.43489959 108.02569767  93.06320016 127.09119795\n",
            "  75.14140033 113.55509908 121.47900041 111.35069881 118.91619888\n",
            " 120.27279924 160.31060037 168.51620089 147.06449674  85.59369876\n",
            "  94.32620022  86.79829933  90.3237999  119.05030078 126.43060051\n",
            " 127.49300013 170.54360079 122.24279927 117.62959844  98.55510053\n",
            " 168.06860158 142.99819816 132.09980232 121.19800249 120.89549925\n",
            " 119.64790093 114.37230131 118.21010033 107.38680102 127.88100036\n",
            " 114.0621996  107.15289997 116.76930037 119.70669871  88.9601005\n",
            "  88.34369882 146.1299022  126.78910015 113.17350037 110.34049843\n",
            " 108.33409912  77.08189908 168.39660184 114.21329907 121.57399938\n",
            " 128.03540157 154.89909821  91.79399966 135.54530085 158.92480383\n",
            " 125.3440006  125.46340044 130.65130101 114.68180071 119.81929957\n",
            "  92.16999961 110.11749911 167.26549965 158.06019867 114.3765997\n",
            " 106.54990112  79.40059997 113.24730066 125.75710085 107.12789965\n",
            " 119.1806013  155.98480288 159.44939933 120.3244001  134.4500024\n",
            " 101.62789996 117.47919801 119.19520031 112.88180075 102.75059929\n",
            " 160.15779789  98.91920033 147.58789923 125.57450114 170.28349947\n",
            " 125.6346993  127.27929753 127.43110165 113.62479943 113.1346007\n",
            " 123.53849909 102.03549896  89.14589964 124.34589922 101.06539931\n",
            " 107.10179967 113.92090066 117.34670086  99.18389956 121.84710054\n",
            " 163.60659964  87.47259883 106.6078996  117.23410057 127.5852014\n",
            " 124.05910082  80.83269909 120.32440061 157.5099982   87.92499947\n",
            " 110.07529965 118.95049912 172.14249914 102.97889886 105.84530001\n",
            " 122.52050047 157.49589778  87.70309815  93.52900017 112.55260046\n",
            " 176.78399926 114.27400004 119.19190013  94.5947008  125.90419992\n",
            " 166.05440018 114.78520032 116.87050117  88.41339902 148.8840007\n",
            " 120.40529937  89.45950005 111.99399996 117.35729981 118.64940112\n",
            "  88.46529954  94.0036998  116.95630046 118.65200177 120.35880083\n",
            " 126.7777982  121.93069963 152.01010042 164.7022006  118.62569975\n",
            " 120.29160153 149.76470051 118.45749925 172.58859947 105.53199931\n",
            " 104.97940094 149.73020111 113.63000071 124.91270094 147.76489901\n",
            " 119.57780101 115.28710044 112.54849993 113.44320195 139.84420093\n",
            " 117.87479771 102.97830042 115.94670105 103.66380164  98.94920037\n",
            " 117.2947009   90.61229989  91.50530088 153.32129875 102.70039963\n",
            " 154.57390113 114.22570155 139.47170121  90.1379977  115.61339941\n",
            " 114.45749983 122.55160026 121.84890017 165.25800212  92.75899948\n",
            " 135.46920178 121.38429879 120.80620072 104.61770014 142.49950356\n",
            " 121.50529903 116.91060067 113.49530121 127.0971974  122.83189941\n",
            " 125.81279949 121.28810017  86.84599938 132.32730111 144.97030211\n",
            "  92.74709948 158.60999965 158.99690266 126.50689884 164.98709962\n",
            " 108.75659958 109.67190064 103.6128981   94.41100063 127.78350288\n",
            " 107.12290043 163.50369972 121.72350055 132.08790076 130.71990146\n",
            " 160.38430046  90.15699805 175.15240136 127.27550085 126.71979854\n",
            "  86.46949934 124.64339988 149.78969723  89.67540032 106.65839979\n",
            " 108.90399983  84.27789913 136.17700021 155.01210249 139.37430394\n",
            "  73.66920014 151.96820164 126.19919988 126.80400001 127.48689897\n",
            " 108.61909949 156.18609935 114.56220091 117.04250138 125.31619929\n",
            " 154.20120192 121.41000021 156.36689873  92.90840064 125.53660142\n",
            " 125.41140015  87.84860072  92.1129991  126.38899933 128.27820349\n",
            " 113.15130038 117.81059775 121.01260022 127.05169838 119.58830118\n",
            " 136.40860106  93.87589912 120.00360052 113.00670104  94.18179919\n",
            " 108.82320018  86.72119921 108.92419969  89.53309979  92.30310021\n",
            " 131.63450256 162.39759987  89.37349971 119.59370105 133.14160202\n",
            " 124.10650036 128.31780182 101.78499836  88.9308989  131.73120091\n",
            " 119.74950022 108.60969949 168.34470141 115.11980032  86.62119881\n",
            " 118.85310064  91.07479968 162.3665006  116.54370058 121.56490031\n",
            " 160.26879816 120.15409918 112.95169925 108.54669887 126.7546999\n",
            "  75.93130044 103.01629981 127.80780301 121.74999893  92.51609981\n",
            " 131.89030059 118.07960131 115.92409977 154.37310307 160.06690112\n",
            " 110.17399974 154.36149831 119.30810078 160.19410146 118.3508\n",
            " 159.89859873 115.21119957 116.71130035 148.93119889 114.77700024\n",
            " 125.70169882 166.61889937 117.61780009 125.0628991  153.23460386\n",
            " 153.53880181 132.19370048 114.7320005  121.20840227 124.92970033\n",
            "  89.92520028 122.94529991 154.60240142 111.71690046 106.65089969\n",
            " 161.179101   118.65749985 165.60719968 134.1385003  114.6636997\n",
            " 152.86129864 168.6308998  115.20270024 113.83570103 159.23249928\n",
            "  85.34609882 127.12520053 127.87030032 128.89789992 124.23680077\n",
            " 123.96730085  90.52530001 153.23000009  97.02579961 137.3211992\n",
            "  88.93369909 107.08969997 115.08710055 112.82900097 124.27959928\n",
            "  91.43929843 125.36120118 162.48359848 119.91009903 165.13900154\n",
            " 126.84309783 112.3976003  127.48449944  95.03529942  90.94809989\n",
            " 103.47839902 120.89389991  83.26909939 126.39300013 160.25880455\n",
            " 117.33720083 118.33549979 120.05159994 122.65709949 120.10970125\n",
            " 121.35699998 118.51820046 107.02310012 148.33840006 126.02039877\n",
            " 115.64010048  74.10169994 127.87350129 154.47420065 122.16299992\n",
            " 125.5646007   88.73970062 103.29539858 124.33380029 120.3184\n",
            "  73.21590085 151.77709972 121.43330034 104.92350016  86.45859769\n",
            " 114.97479927 172.23989787 119.96450031 161.26909781 113.21289946\n",
            " 121.08260043 118.39700098  95.89769979 118.68090061 126.03180001\n",
            " 118.48069952  95.98520085 153.98010198 122.19850021 147.80279994\n",
            " 159.71120216 113.74970019 122.4004996  148.09289741 127.0505004\n",
            " 165.92670108 135.46270082 120.1067993  167.29919853 108.29929945\n",
            " 121.79029815 138.81690125 106.38239875]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fu7A1hhMhqa9",
        "outputId": "3609de12-2ab2-46fd-c137-7e3777ddd8be"
      },
      "source": [
        "# R squared error\n",
        "error_score = metrics.r2_score(Y_test, test_data_prediction)\n",
        "print(\"R squared error : \", error_score)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "R squared error :  0.9887338861925125\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "f1fiqOMkiZNL"
      },
      "source": [
        "Compare the Actual Values and Predicted Values in a Plot"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QoC4g_tBiE4A"
      },
      "source": [
        "Y_test = list(Y_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "sMSVMVtFijxo",
        "outputId": "34404933-1a9f-4e34-93f2-790c9665cad7"
      },
      "source": [
        "plt.plot(Y_test, color='blue', label = 'Actual Value')\n",
        "plt.plot(test_data_prediction, color='green', label='Predicted Value')\n",
        "plt.title('Actual Price vs Predicted Price')\n",
        "plt.xlabel('Number of values')\n",
        "plt.ylabel('GLD Price')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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n/0W/375oh7rUbXYmu9St8KtvVydUIhSIYwc2sixY79I40v/yeALq+5eF0H5Xnf9Cm/z5s20trZWlJqhv2P77WHLlur19OXcX3tqJIUQNDU1sWbNGs/2WbMijncMzT3LFIZunsl7haU9+pt+qHtSkpLdZmTZbUb3C+3OIhUKVSKs9nIl8JeRVBBCcMPnftmZrnU5ukMoQGW5erYGmGbX5f1x1BIx9cG7E/57t2kTkQn+VEGqbA97H63+9kOsb1vfI7/lmpP998Pd01+QGpqrRCHXOaYQGx3ZR+Cqj7ZeXXB/RSFjeSAMyrlR270pYwuFGKHgqI969jmqydYwrK6nvA1c24F3a+8K72qQCoUq4TCFLrYp9C143R+3Rtxyyy0IIXj22WBOHD8uvfRSNm7cWPVvXX311Zx99tmebatWrWL06NGUfAEB06dP57HHwlM5rFq1it132Z2pLVNprm3W9vTNZ0lNjH0pTqGr4RqUg3us7alQ2OpR4zCFKtVH/oLzfRBNds60USO3XqYwf/589t57b+bPn1/22M4KhTC0trYyduxYHn74YWfbs88+y0cffcTuu+8ee27OzPULFVxv2RS6DM8dXv6YiNsgXDtzv8HWK7q7GW7xnCoNzX09fy4weLDgjTXQ0HV55UJx/vnw8GIrNmFQrmvanD4dLr00/pj169fz97//nYULF3LEEUfwrW99C4BiscjXvvY17r77bgzD4LTTTkNKyRtvvMF+++1Hc3MzCxcupL6+nvXrLZ31DTfcwB133MHVV1/N7bffzne+8x3a2tpoamrij3/8Iy0tLZH9mDNnDgsWLODjH/84AAsWLGD27NmsWrWKuXPnsmGD5Zn185//nD333NNz7tVXX82dD97JV7/7VYSAww8/nC9/+cvsu+++3HPPPVx88cVs2bKF7bbbjt///vfU19dXO6RVQzHNfssU5t9muZomcA+Psin0I6KQMoVqEVZmsxI4QWp9WH3UH/WhleDWW2/l0EMPZYcddqCpqYl//etfAFx55ZWsWrWKJ598kqeeeooTTzyRc889l5EjR7Jw4UIWLlwY2+7ee+/No48+yhNPPMHs2bO55JJLYo8//vjjueWWW+josBYY119/PXPmzGH48OHce++9PP7441x//fWce+65se3oePfdd/nOd77Dfffdx+OPP87MmTP5yU9+kvj8rkRfTHNRCY48UjjlWKMQ9a44ZuZ+JBV65S4JIS4ATsViVU8DnwNGAAuAJuBfwFwpZVu3deLKxVD7blXBYQA1EWU2k8LsB+ojhe5+oC+9FJa+8RwAM0fO7Nbf0jF//nzOO+88AGbPns38+fOZMWMG9913H2eccQYZexIbOrSyxHqrV6/mhBNO4M0336Strc315Y9AS0sLU6ZM4f7776elpYVMJsOUKVNYt24dZ599Nk8++SSmafL888/HtqNn6X300UdZsWIFe+21FwBtbW3MivIZ7WY4wWv9VH10yy3lV/phGZJB99JKhUIkhBCjgHOBSVLKTUKIPwOzgcOAn0opFwghfgV8Hriiu/rx9ZN2ZcWK6s93ymxWa2juhnxCKZLjvffe44EHHuDpp59GCEGxWEQIwQ9/+MPEbej6/M2bNzufzznnHC688EKOPPJIFi1axLx588q2pVRILS0tzJkzB4Cf/vSntLS0sGzZMkqlEoWQnCeZTMY1Ugu3H1JKDjrooES2ku6G6OfqIyGSe3YFF1Bumov+gt6amTJAjRAiA9QCbwL7AzfY+68BjurODvzf/8Gtt1Z/vlMnoUr1UaYfeB+pSa8n1EfNxvYMM7fv9t9RuOGGG5g7dy6vvPIKq1at4rXXXmPcuHE8/PDDHHTQQfz617921DnvvfceAIMGDeKjjz5y2mhpaWHlypWUSiVuvvlmZ/u6desYNcpKZ3DNNdck6s8xxxzDnXfeyfXXX8/s2bOddkaMGIFhGPzhD3+gWAw+a62trTz/zPOUSiVWv/Yaixdb2Tn32GMP/vGPf/Diiy8CsGHDhrJMo7ug0kn39YL1nUEkU0i9j8pDSvk68CPgVSxhsA5LXfSBlFItu1cDoUlChBBfEEIsFUIs9UdU9iQ6yxQqKZ3YW+jJB7p1m8Fs29L1NZWjMH/+fI725QI59thjmT9/Pqeeeipjx45l6tSpTJs2jT/96U8AfOELX+DQQw9lv/2slKzf//73Ofzww9lzzz0ZMcLNPDpv3jyOO+44ZsyYQXNzM0kwePBgZs2aRUtLC+PHW9lgzzzzTK655hqmTZvGs88+S11I2te99tqLkWNHcvy+x3PBBeezyy67ADBs2DCuvvpq5syZw9SpU5k1a1Yit9vugLIp5PspU6gIW4FLKlLKHv0DhgAPAMOALHAL8FngRe2YMcDycm3NmDFD9hb+uXy1ZB7S+Orwqs5fsvItyTwk36zr4p6Fg3lI5lHROWfcfoZkHvKVD17p8v6sWLGiy9scqFjy+hK55PUlsq29o0d/N+k9nHDhFyTzkC+9tqGbe9R72Pmr50jmIY+95DLP9olfPk0yD3niT3/dSz0LB7BURsyrvaE+OhB4WUq5RkrZDtwE7AUMttVJAKOB13uhb4lRV7DVR0YnXVL7sProsk9cxrIzljklNVP0bfTVmAXVr3x262UKrvooghH0I6bQG0LhVWAPIUStsJ6WA4AVwELg0/YxJwOd0Ph3PzptaO4HcQo5MxcoWJ+i76JvigTXppDfim0KCgEzcw/a5boKvWFTeAzLoPw4ljuqAVwJfA24UAjxIpZb6lU93bdKkM/ZD3i1LqnK+6ivvskp+h/66LPk5D7K9P2FUNVQYUdRNoV+JBR6hc9JKS8mGCHwErBbL3SnKtTlLPfAzPOfLnNkOMx+ELyWIkVXwBAGlEwyma33WY8KUuuP3kdbr5KvmzGoLguXvMPJJ1bnMeOqf3voRbnuLjDbqg7WS9H3EeUW2dsQwoBShq248FpkkFoqFAYQsll4b/UwBlWZF8gxKfQUU3jx0J75nRS9h74pE6yJsWRu1UIhavDd9Bf9B1uxkq/7MWRIdA75cugHduatHqZpMn36dKZMmcJxxx3XqQyop5xyCjfcYMVennrqqayICZdftGgRjzzySMW/0drayrvvvuvZ9rnPfY6b/nAT4E5Lt9xyC5/4xCcS9bUn0Nr2SVhy1lYuFCwEbAeOraH/iIV0auol5DN5ADIvf7KXezJwUVNTw5NPPsny5cvJ5XL86le/8uxXEc2V4re//S2TJk2K3F+tUAjDnDlzuOfWewBXhbFgwQInVUZfwLbFAzHuv6RXiwB1N8pFNPcnpOqjXkJdrgZ+uooaMaL8wVs5zr/7fJ5868kubXP6NtO59NAyubM1fOxjH+Opp55i0aJFXHTRRQwZMoRnn32WlStX8vWvf51FixaxZcsWzjrrLE4//XSklJxzzjnce++9jBkzhlzOzfm977778qMf/YiZM2dy9913881vfpNisUhzczNXXXUVv/rVrzBNk+uuu47LL7+cHXfckTPOOINXX30VsOo27LXXXqxdu5Y5c+bw+uuvM2vWrNDV5gEHHMCquat49+13YaSVzuK+++7jyiuv5Nvf/ja33347mzZtYs899+TXv/51IJahtbWVpUuX0tzczNKlS/nyl7/MokWL2LBhA+eccw7Lly+nvb2defPm8alPfaqqe2EYDAiWANFeRilTSFEWQgDrtkWUuqiAQBkccQTY2RlS+NDR0cFdd93FzjvvDMDjjz/OZZddxvPPP89VV11FY2MjS5YsYcmSJfzmN7/h5Zdf5uabb+a5555jxYoVXHvttaEr/zVr1nDaaadx4403smzZMv7yl7/Q2trKGWecwQUXXMCTTz7Jxz72Mc477zwuuOAClixZwo033sipp54KwLe+9S323ntvnnnmGY4++mhHaOgwTZP9D9ufe2+/F4Dbb7+dfffdl4aGBs4++2yWLFnC8uXL2bRpE3fccUfiMfnud7/L/vvvz+LFi1m4cCFf+cpXnLoOlUKIrV8ojOzYG4DRxgzP9v4Yp5AyhV5CTy8cbrutZ3+vElSyou9KbNq0ienTpwMWU/j85z/PI488wm677eaku77nnnt46qmnHB38unXreOGFF3jooYeYM2cOpmkycuRI9t9//0D7jz76KPvss4/TVlQK7vvuu89jg/jwww9Zv349Dz30EDfdZNkLPvnJTzJkyJDQ8484/ER++L2L4H8s1dHcuXMBWLhwIZdccgkbN27kvffeY/LkyRxxxBGJxuaee+7htttu40c/+hFgZV999dVX2WmnnRKdr6O2FkLSNm1VmNBxDPzwLbb/X28xJaU+KvUjppAKhV6Ceka2Zj1rX4eyKfihJ56TUnL55ZdzyCGHeI658847u6wfpVKJRx99NDQ1dhKcePSRXPSl81i2bBmPPPIICxYsYPPmzZx55pksXbqUMWPGMG/ePE96bwU99ba+X0rJjTfeyMSJE6u7KA3nnQdVap76Fza0BBZ70zddwFMf3c8uE2b3Tp+qQKo+6iUo76PGxt7tR4p4HHLIIVxxxRW0t7cD8Pzzz7Nhwwb22Wcfrr/+eorFIm+++WZoNbY99tiDhx56iJdffhmITsF98MEHc/nllzvflaDaZ599nAytd911F++//35oHw1DcMIJJ3DyySfziU98gkKh4Ezwzc3NrF+/PtLbqLW11ak4d+ONN3qu+/LLL3d04U888US5oYpESwuUKTfd7+HUYvYJhUY5Dn75DINEdDnWvoZUKPQSGhvhJz+BBx7o7Z6kiMOpp57KpEmT2GWXXZgyZQqnn346HR0dHH300UyYMIFJkyZx0kknhVY1GzZsGFdeeSXHHHMM06ZN44QTTgDgiCOO4Oabb2b69Ok8/PDD/OxnP2Pp0qVMnTqVSZMmOV5QF198MQ899BCTJ0/mpptuYuzY6MSEc+bMYdmyZY7X0eDBgznttNOYMmUKhxxyCLvuumvoeRdffDHnnXceM2fOxNQU/xdddBHt7e1MnTqVyZMnc9FFF1U9hgMBiuRls97tF15o1Qv/zGd6vk/VQvQnq7gfM2fOlEuXLu3tbqSoAitXrqxKP52i7yC9hy7Wr4dvf9v6q1IL2KMQQvxLShla+za1KaRIkSJFJ1FfD5dc0tu96Bqk6qMUKVKkSOEgFQopeg39WXU50JHeu60XqVBI0SsoFAqsXbs2nVz6IaSUrF27tmoX2hR9G6lNIUWvYPTo0axevZo1a9b0dldSVIFCocDo0aN7uxspugGpUEjRK8hms06kb4oUKfoOUvVRihQpUqRwkAqFFClSpEjhIBUKKVKkSJHCQb+OaBZCrAFeqfL0ZuDdskcNDKRj4SIdCwvpOLjYGsdiWynlsLAd/VoodAZCiKVRYd4DDelYuEjHwkI6Di4G2lik6qMUKVKkSOEgFQopUqRIkcLBQBYKV/Z2B/oQ0rFwkY6FhXQcXAyosRiwNoUUKVKkSBHEQGYKKVKkSJHCh1QopEiRIkUKBwNSKAghDhVCPCeEeFEI8fXe7k93QwjxOyHEO0KI5dq2oUKIe4UQL9j/h9jbhRDiZ/bYPCWE2KX3et61EEKMEUIsFEKsEEI8I4Q4z94+EMeiIIRYLIRYZo/Ft+zt44QQj9nXfL0QImdvz9vfX7T3t/Zm/7saQghTCPGEEOIO+/uAHAcYAVKTCgAABndJREFUgEJBCGECvwA+AUwC5gghJvVur7odVwOH+rZ9HbhfSjkBuN/+Dta4TLD/vgBc0UN97Al0AF+SUk4C9gDOsu/9QByLLcD+UsppwHTgUCHEHsAPgJ9KKbcH3gc+bx//eeB9e/tP7eO2JpwHrNS+D9RxsHKjD6Q/YBbwN+37N4Bv9Ha/euC6W4Hl2vfngBH25xHAc/bnXwNzwo7b2v6AW4GDBvpYALXA48DuWJG7GXu7864AfwNm2Z8z9nGit/veRdc/GmsxsD9wByAG4jiovwHHFIBRwGva99X2toGGFinlm/bnt4AW+/OAGB+b9v8H8BgDdCxslcmTwDvAvcC/gQ+klB32Ifr1OmNh718HNPVsj7sNlwJfBUr29yYG5jgAA1B9lCIIaS17BoxvshCiHrgROF9K+aG+byCNhZSyKKWcjrVS3g3YsZe71OMQQhwOvCOl/Fdv96WvYCAKhdeBMdr30fa2gYa3hRAjAOz/79jbt+rxEUJksQTCH6WUN9mbB+RYKEgpPwAWYqlJBgshVPEt/XqdsbD3NwJre7ir3YG9gCOFEKuABVgqpMsYeOPgYCAKhSXABNu7IAfMBm7r5T71Bm4DTrY/n4ylX1fbT7I9b/YA1mmqlX4NIYQArgJWSil/ou0aiGMxTAgx2P5cg2VbWYklHD5tH+YfCzVGnwYesFlVv4aU8htSytFSylasueABKeWJDLBx8KC3jRq98QccBjyPpUP9r97uTw9c73zgTaAdSz/6eSw96P3AC8B9wFD7WIHlnfVv4GlgZm/3vwvHYW8s1dBTwJP232EDdCymAk/YY7Ec+B97+3hgMfAi8Bcgb28v2N9ftPeP7+1r6IYx2Re4Y6CPQ5rmIkWKFClSOBiI6qMUKVKkSBGBVCikSJEiRQoHqVBIkSJFihQOUqGQIkWKFCkcpEIhRYoUKVI4SIVCin4DIYQUQvxY+/5lIcS8Lmr7aiHEp8sf2enfOU4IsVIIsbCT7bTqWW9TpOgqpEIhRX/CFuAYIURzb3dEhxb5mgSfB06TUu7XXf1JkaIzSIVCiv6EDqx6uRf4d/hX+kKI9fb/fYUQDwohbhVCvCSE+L4Q4kS7lsDTQojttGYOFEIsFUI8b+fEUUnjfiiEWGLXVDhda/dhIcRtwIqQ/syx218uhPiBve1/sALorhJC/NB3/AIhxCf912MzgoeFEI/bf3uG/NYpQoifa9/vEELsa38+WAjxT/vcv9h5n7DHYYV9TT8qO/IpBgwqWeGkSNEX8AvgKSHEJRWcMw3YCXgPeAn4rZRyN2EV2TkHON8+rhUrMdx2wEIhxPbASVjpLXYVQuSBfwgh7rGP3wWYIqV8Wf8xIcRIrDz7M7By8d8jhDhKSvltIcT+wJellEt9fbweOB74q51+5QDgi1hR1QdJKTcLISZgRafPTHLRNqP6b+BAKeUGIcTXgAuFEL8AjgZ2lFJKle4iRQpIhUKKfgYp5YdCiGuBc4FNCU9bIu2cRUKIfwNqUn8a0NU4f5ZSloAXhBAvYWUNPRiYqrGQRqyiO23AYr9AsLErsEhKucb+zT8C+wC3xPTxLuAyW/AcCjwkpdwkhGgEfi6EmA4UgR0SXjNYhYQmYQkygBzwT6x0z5uxGMsdWDUEUqQAUqGQon/iUqyiML/XtnVgq0OFEAbWBKiwRftc0r6X8L4D/pwvEmulfo6U8m/6Dls9s6G67gdhM4FFwCHACVgZO8FSlb2NxXYMrMncD+fabRRUN4F7pZRz/CcIIXbDYiOfBs7Gyg6aIkVqU0jR/yClfA/4M26JRIBVWOoagCOBbBVNHyeEMGw7w3isSmt/A75op9xGCLGDEKKuTDuLgY8LIZqFVf51DvBggt+/Hvgc8DHgbntbI/CmzWDmAmbIeauA6Xbfx2CpwAAeBfay1WAIIers/tcDjVLKO7GEzrQEfUsxQJAyhRT9FT/GWuEq/Aa4VQixDGtCrWYV/yrWhN4AnGGv3n+LZWt43E69vQY4Kq4RKeWbQoivY6VfFsBfpZS3xp1j4x7gD8CtUso2e9svgRuFECcRfV3/AF7GMnivxGJRSCnXCCFOAebbaimwbAwfYY1Vwe7fhQn6lmKAIM2SmiJFihQpHKTqoxQpUqRI4SAVCilSpEiRwkEqFFKkSJEihYNUKKRIkSJFCgepUEiRIkWKFA5SoZAiRYoUKRykQiFFihQpUjj4f9WLeofLWPz2AAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}