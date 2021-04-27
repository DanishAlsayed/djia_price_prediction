import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from tqdm import tqdm
matplotlib.use('Agg')

PATH = os.path.dirname(__file__)
GAF = os.path.join(PATH, 'images')
TRAIN_PATH = os.path.join(GAF, 'train')
TRAIN_BUY = os.path.join(TRAIN_PATH, 'BUY')
TRAIN_SELL = os.path.join(TRAIN_PATH, 'SELL')
MODELS_PATH = os.path.join(PATH, 'models')
PLOT_PATH = os.path.join(PATH, 'plots')

def data_to_gaf(df):

    # Check if directories is empty
    if len(os.listdir(TRAIN_BUY)) == 0 and len(os.listdir(TRAIN_SELL)) == 0:
        print('PROCESSING DATA')
        df['DateTime'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df['Close'] = df['Close'].replace(to_replace=0, method='ffill')
        set_gaf_data(df)
        print('DONE!')



def create_dir(path):
    """
    :param path: String
    :return: None
    """
    if not os.path.exists(path):
        os.mkdir(path)

def create_all_dir():
    # Creates directories and sub directories needed for the project to run
    """

    :param GAF: String
    :param TRAIN_PATH: String
    :param TRAIN_BUY: String
    :param TRAIN_SELL: String
    :param MODELS_PATH: String
    :return:
    """
    create_dir(GAF)
    create_dir(TRAIN_PATH)
    create_dir(TRAIN_BUY)
    create_dir(TRAIN_SELL)
    create_dir(MODELS_PATH)
    create_dir(PLOT_PATH)


def set_gaf_data(df):
    """
    :param df: DataFrame data_slice
    :return: None
    """
    dates = df['DateTime'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = 20
    timeframe = 20
    # Container to store data_slice for the creation of GAF
    decision_map = {key: [] for key in ['BUY', 'SELL']}
    while True:
        if index >= len(list_dates) - 1:
            break

        # Select appropriate timeframe
        data_slice = df.loc[(df['DateTime'] > list_dates[index - timeframe]) & (df['DateTime'] < list_dates[index])]
        gafs = data_slice['Close'].tail(timeframe)

        # Decide what trading position we should take on that day
        future_value = df[df['DateTime'].dt.date.astype(str) == list_dates[index]]['Close'].iloc[-1]
        current_value = data_slice['Close'].iloc[-1]
        decision = trading_action(future_close=future_value, current_close=current_value)
        decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    print('GENERATING IMAGES')
    # Generate the images from processed data_slice

    generate_gaf(decision_map)

    # Log stuff
    dt_points = dates.shape[0]
    total_sell = len(decision_map['SELL'])
    total_buy = len(decision_map['BUY'])
    images_created = total_sell + total_buy
    print("========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
          "\nTotal BUY positions: {2}\nTotal SELL positions: {3}".format(dt_points,
                                                                           images_created,
                                                                           total_sell,
                                                                           total_buy))


def trading_action(future_close, current_close):
    """
    :param future_close: Integer
    :param current_close: Integer
    :return: Folder destination as String
    """
    current_close = current_close
    future_close = future_close
    if current_close < future_close:
        decision = 'BUY'
    else:
        decision = 'SELL'
    return decision


def generate_gaf(images_data):
    """
    :param images_data:
    :return:
    """
    for decision, data in images_data.items():
        for image_data in tqdm(data):
            to_plot = [create_gaf(image_data[1])['gadf']]
            create_images(X_plots=to_plot,
                              image_name='{0}'.format(image_data[0].replace('-', '_')),
                              destination=decision)


# Pass times-eries and create a Gramian Angular Field image
# Grab times-eries and draw the charts
def create_gaf(ts):
    """
    :param ts:
    :return:
    """
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    return data


# Create images of the bundle that we pass
def create_images(X_plots, image_name, destination, image_matrix=(1, 1)):
    """
    :param X_plots:
    :param image_name:
    :param destination:
    :param image_matrix:
    :return:
    """
    fig = plt.figure(figsize=[img * 4 for img in image_matrix])
    grid = ImageGrid(fig,
                     111,
                     axes_pad=0,
                     nrows_ncols=image_matrix,
                     share_all=True,
                     )
    images = X_plots
    for image, ax in zip(images, grid):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='rainbow', origin='lower')

    repo = os.path.join(TRAIN_PATH, destination)
    fig.savefig(os.path.join(repo, image_name))
    plt.close(fig)


