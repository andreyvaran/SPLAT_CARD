from catboost import CatBoostRegressor

from scipy.interpolate import LinearNDInterpolator

import numpy as np

import os

from skimage import color

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

import math

from datetime import datetime

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

import time


def distance(a, b):

    return np.sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5


def vect_length(a):

    return np.sum([i * i for i in a])


def rgblab(l):

    return np.array(color.rgb2lab([[i[0] / 255, i[1] / 255, i[2] / 255] for i in l]))


def labrgb(l):

    return 255 * np.array(color.lab2rgb([[i[0], i[1], i[2]] for i in l]))


def rgbhsv(l):

    # temp_l = np.array(color.rgb2hsv(np.array([[i[0]/255, i[1]/255, i[2]/255] for i in l])))

    # for i in range(len(temp_l)):

    # print('rgb:', l[i], 'hsv: ', temp_l[i])

    return np.array(
        color.rgb2hsv(np.array([[i[0] / 255, i[1] / 255, i[2] / 255] for i in l]))
    )


def hsvrgb(l):

    # temp_l = 255*np.array(color.hsv2rgb(np.array([[i[0], i[1], i[2]] for i in l])))

    # for i in range(len(temp_l)):

    # print('hsv:', l[i], 'rgb: ', temp_l[i])

    return 255 * np.array(color.hsv2rgb(np.array([[i[0], i[1], i[2]] for i in l])))


def rgbxyz(l):

    return np.array(color.rgb2xyz([[i[0] / 255, i[1] / 255, i[2] / 255] for i in l]))


def xyzrgb(l):

    return 255 * np.array(color.xyz2rgb([[i[0], i[1], i[2]] for i in l]))


def rgbluv(l):

    return np.array(color.rgb2luv([[i[0] / 255, i[1] / 255, i[2] / 255] for i in l]))


def luvrgb(l):

    return 255 * np.array(color.luv2rgb([[i[0], i[1], i[2]] for i in l]))


def to_rgb(l, mode):

    if mode == "LAB":

        return labrgb(l)

    if mode == "XYZ":

        return xyzrgb(l)

    if mode == "HSV":

        return hsvrgb(l)

    if mode == "LUV":

        return luvrgb(l)


class gbtree_vf:

    def __init__(self, ref, params, mode):

        self.ref = ref

        self.interpolated_points = []

        self.interpolated_values = []

        self.params = params

        self.catboostmodel = CatBoostRegressor(**params)

        self.mode = mode

        self.dim_change = False

    def extra_dim(self):

        self.LAB_X = np.array(rgblab(self.X))

        self.LAB_Y = np.array(rgblab(self.Y))

        self.LX = np.array([i[0] for i in self.LAB_X])

        self.LY = np.array([i[0] for i in self.LAB_Y])

        self.LX = self.LX.reshape((self.LX.shape[0], 1))

        self.LY = self.LY.reshape((self.LY.shape[0], 1))

        self.X = np.concatenate((self.X, self.LX), axis=1)

        self.Y = np.concatenate((self.Y, self.LY), axis=1)

        print(self.X.shape)

        print(self.X[0])

    def interpolate(self):

        interps = []

        for j in range(len(self.vector_field[0])):

            temp_interp = LinearNDInterpolator(
                self.input, [i[j] for i in self.vector_field]
            )

            interps.append(temp_interp)

        distances = []

        for i in range(len(self.input)):

            for j in range(i + 1, len(self.input)):

                temp_dist = distance(self.input[i], self.input[j])

                distances.append([temp_dist, i, j])

        distances = sorted(distances, key=lambda x: x[0])

        minDist = np.min([element[0] for element in distances])

        for el in distances:

            if el[0] < 16 * minDist:

                step = el[0] / 2

            else:

                step = el[0] / 3

            i = el[1]

            j = el[2]

            dist = distance(self.input[i], self.input[j])

            vect_dir = (np.array(self.input[j]) - np.array(self.input[i])) / dist

            s = step

            while (s < dist) and distance(
                self.input[j], np.array(self.input[i]) + np.array(vect_dir) * s
            ) > minDist / 2:

                c = np.array(self.input[i]) + np.array(vect_dir) * s

                cs = []

                for ax in range(len(interps)):

                    temp_c = interps[ax].__call__(c)

                    cs.append(temp_c)

                if not all(v is None for v in cs) and not all(
                    math.isnan(v) for v in cs
                ):

                    temp = np.array(cs)

                    temp = np.ravel(temp)

                    self.interpolated_points.append(c)

                    self.interpolated_values.append(temp)

                s += step

    def vect_field(self, space):

        self.input = space

        self.vector_field = np.array(self.ref) - np.array(self.input)

        self.interpolate()

        self.input = np.array(self.input)

        self.vector_field = np.array(self.vector_field)

        self.interpolated_points = np.array(self.interpolated_points)

        self.interpolated_values = np.array(self.interpolated_values)

        self.X = np.concatenate((self.input, self.interpolated_points), axis=0)

        self.Y = np.concatenate((self.vector_field, self.interpolated_values), axis=0)

    def normalize(self):

        self.norm_X = MinMaxScaler()

        self.norm_Y = MinMaxScaler()

        self.norm_X.fit(self.X)

        self.norm_Y.fit(self.Y)

        self.x = self.norm_X.transform(self.X)

        self.y = self.norm_Y.transform(self.Y)

    def fit(self, space, extra_dimensions=False):

        self.vect_field(space)

        if extra_dimensions == True:

            self.extra_dim()

            self.dim_change = True

        self.normalize()

        if self.dim_change:

            self.params["per_float_feature_quantization"] = "3:border_count=1024"

            self.catboostmodel = CatBoostRegressor(**self.params)

            self.catboostmodel.fit(self.x, self.y, silent=True)

        else:

            self.catboostmodel = CatBoostRegressor(**self.params)

            self.catboostmodel.fit(self.x, self.y, silent=True)

    def transform(self, c):

        # print(c)

        c_original = c

        if self.dim_change:

            lab_c = rgblab([c])

            c_temp = c

            c_temp.append(lab_c[0][0])

            inp = self.norm_X.transform([c_temp])

        else:

            inp = self.norm_X.transform([c])

        temp = self.catboostmodel.predict(inp)

        temp = np.array(self.norm_Y.inverse_transform(np.array(temp)))

        temp = temp[0]

        if self.dim_change == False:

            return temp + np.array(c_original)

        else:

            return np.array([temp[i] for i in range(3)]) + np.array(
                [c_original[i] for i in range(3)]
            )


class cc_mdl_ft:

    def __init__(self, y_train, y_test):

        self.y_train = np.array(y_train)

        self.y_test = np.array(y_test)

        self.y = np.concatenate((self.y_train, self.y_test))

        path = os.getcwd()

        curDatetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        self.savepath = os.path.join(path, "CC" + curDatetime)

    def fit(
        self,
        x_train,
        x_test,
        ft=True,
        params={
            "iterations": 2500,
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": 0.15,
            "depth": 4,
            "l2_leaf_reg": 2,
        },
    ):

        self.x_train = np.array(x_train)

        self.x_test = np.array(x_test)

        self.x = np.concatenate((self.x_train, self.x_test))

        if not ft:

            self.model = gbtree_vf(self.y, params, 255)

            self.model.fit(self.x)

            self.transformed_x = [self.transform(el_x) for el_x in self.x]

            return params

        else:

            for iteration_numb in [3500]:

                for lr in [0.2, 0.25, 0.15, 0.1]:

                    for d in [8, 10, 12]:

                        for l2 in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]:

                            params = {
                                "iterations": iteration_numb,
                                "loss_function": "MultiRMSE",
                                "eval_metric": "MultiRMSE",
                                "learning_rate": lr,
                                "depth": d,
                                "l2_leaf_reg": l2,
                            }

                            self.best_rmse = [-1]

                            self.best_mae = [-1]

                            self.best_mape = [-1]

                            self.model = gbtree_vf(self.y_train, params, 255)

                            self.model.fit(self.x_train)

                            self.transformed_x_train = [
                                self.transform(el_x) for el_x in self.x_train
                            ]

                            self.transformed_x_test = [
                                self.transform(el_x) for el_x in self.x_test
                            ]

                            rmse_res_train = np.sqrt(
                                mean_squared_error(
                                    self.y_train, self.transformed_x_train
                                )
                            )

                            mape_res_train = mean_absolute_percentage_error(
                                self.y_train, self.transformed_x_train
                            )

                            mae_res_train = mean_absolute_error(
                                self.y_train, self.transformed_x_train
                            )

                            rmse_res_test = np.sqrt(
                                mean_squared_error(self.y_test, self.transformed_x_test)
                            )

                            mape_res_test = mean_absolute_percentage_error(
                                self.y_test, self.transformed_x_test
                            )

                            mae_res_test = mean_absolute_error(
                                self.y_test, self.transformed_x_test
                            )

                            print("\n \n \n \n \n \n")

                            print(f"iter {iteration_numb}")

                            print(f"lr {lr}")

                            print(f"depth {d}")

                            print(f"l2 {l2}")

                            print("\n \n \n \n \n \n")

                            print(
                                f"Train: rmse {round(rmse_res_train, 3)} mape {round(mape_res_train, 3)} mae{round(mae_res_train, 3)}"
                            )

                            print()

                            print(
                                f"Test: rmse {round(rmse_res_test, 3)} mape {round(mape_res_test, 3)} mae{round(mae_res_test, 3)}"
                            )

                            print("\n \n \n \n \n \n")

                            if (
                                rmse_res_test < self.best_rmse[0]
                                or self.best_rmse[0] == -1
                            ):

                                self.best_rmse = [
                                    rmse_res_test,
                                    iteration_numb,
                                    lr,
                                    d,
                                    l2,
                                ]

                            if (
                                mae_res_test < self.best_mae[0]
                                or self.best_mae[0] == -1
                            ):

                                self.best_mae = [
                                    mae_res_test,
                                    iteration_numb,
                                    lr,
                                    d,
                                    l2,
                                ]

                            if (
                                mape_res_test < self.best_mape[0]
                                or self.best_mape[0] == -1
                            ):

                                self.best_mape = [
                                    mape_res_test,
                                    iteration_numb,
                                    lr,
                                    d,
                                    l2,
                                ]
            print()

            print()

            print(self.best_mae)

            print()

            params = {
                "iterations": self.best_mae[1],
                "loss_function": "MultiRMSE",
                "eval_metric": "MultiRMSE",
                "learning_rate": self.best_mae[2],
                "depth": self.best_mae[3],
                "l2_leaf_reg": self.best_mae[4],
            }

            self.model = gbtree_vf(self.y, params, 255)

            self.model.fit(self.x)

            self.transformed_x = [self.transform(el_x) for el_x in self.x]

            return params

    def transf_image(self, img):

        img = np.array(img)

        print(img.shape)

        img_tr = np.zeros_like(img)

        for i in range(len(img)):

            if i % 50 == 0:

                print(f"Line {i}")

            for j in range(len(img[i])):

                img_tr[i][j] = self.transform(img[i][j])

        return img_tr

    def transform(self, c):

        return self.model.transform(c)

    def train_res(self):

        res = []

        res.append("Y | X_tranformed | X")

        for i in range(len(self.x)):

            res.append(
                str([int(el) for el in self.y[i]])
                + " "
                + str([int(el) for el in self.transformed_x[i]])
                + " "
                + str([int(el) for el in self.x[i]])
            )

        return res

    def metric_calc(self):

        rmse_res = np.sqrt(mean_squared_error(self.y, self.transformed_x))

        mape_res = mean_absolute_percentage_error(self.y, self.transformed_x)

        mae_res = mean_absolute_error(self.y, self.transformed_x)

        return (
            f"Train RMSE {rmse_res}, MAPE {mape_res}, MAE {mae_res}",
            rmse_res,
            mape_res,
            mae_res,
        )

    def visualise(self, x_test, path, y_test=False):

        savepath = os.path.join(path, "CC" + "test")

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[i[0] for i in self.transformed_x],
                    y=[i[1] for i in self.transformed_x],
                    z=[i[2] for i in self.transformed_x],
                    mode="markers",
                    name="x_train_transformed",
                    marker=dict(
                        size=4,
                        color="magenta",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.x],
                    y=[i[1] for i in self.x],
                    z=[i[2] for i in self.x],
                    mode="markers",
                    name="x_train",
                    marker=dict(
                        size=4,
                        color="aqua",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.y],
                    y=[i[1] for i in self.y],
                    z=[i[2] for i in self.y],
                    mode="markers",
                    name="y_train",
                    marker=dict(
                        size=5,
                        color="mediumspringgreen",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[i[0] for i in x_test],
                    y=[i[1] for i in x_test],
                    z=[i[2] for i in x_test],
                    mode="markers",
                    name="test_x",
                    marker=dict(
                        size=7,
                        color="lightskyblue",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[self.transform(i)[0] for i in x_test],
                    y=[self.transform(i)[1] for i in x_test],
                    z=[self.transform(i)[2] for i in x_test],
                    mode="markers",
                    name="test_x_transsform",
                    marker=dict(
                        size=5,
                        color="black",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
            ]
        )

        if y_test:

            fig.add_trace(
                go.Scatter3d(
                    x=[i[0] for i in y_test],
                    y=[i[1] for i in y_test],
                    z=[i[2] for i in y_test],
                    mode="markers",
                    name="y_test",
                    marker=dict(
                        size=5,
                        color="gold",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                )
            )

        fig.update_layout(scene=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B"))

        fig.write_html(savepath)


# -------------------------------------------------------------------------------------------------------


class cc_mdl:

    def __init__(self, y):

        self.y = y

        path = os.getcwd()

        curDatetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        self.savepath = os.path.join(path, "CC" + curDatetime)

    def fit(
        self,
        x,
        params={
            "iterations": 2500,
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": 0.15,
            "depth": 4,
            "l2_leaf_reg": 1.75,
        },
    ):

        self.x = x

        self.model = gbtree_vf(self.y, params, 255)

        self.model.fit(self.x)

        self.transformed_x = [self.transform(el_x) for el_x in self.x]

    def transform(self, c):

        return self.model.transform(c)

    def train_res(self):

        res = []

        res.append("Y | X_tranformed | X")

        for i in range(len(self.x)):

            res.append(
                str([int(el) for el in self.y[i]])
                + " "
                + str([int(el) for el in self.transformed_x[i]])
                + " "
                + str([int(el) for el in self.x[i]])
            )

        return res

    def metric_calc(self):

        rmse_res = np.sqrt(mean_squared_error(self.y, self.transformed_x))

        mape_res = mean_absolute_percentage_error(self.y, self.transformed_x)

        mae_res = mean_absolute_error(self.y, self.transformed_x)

        return (
            f"Train RMSE {rmse_res}, MAPE {mape_res}, MAE {mae_res}",
            rmse_res,
            mape_res,
            mae_res,
        )

    def visualise(self, x_test, path, y_test=False):

        savepath = os.path.join(path, "CC" + "test")

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[i[0] for i in self.transformed_x],
                    y=[i[1] for i in self.transformed_x],
                    z=[i[2] for i in self.transformed_x],
                    mode="markers",
                    name="x_train_transformed",
                    marker=dict(
                        size=4,
                        color="magenta",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.x],
                    y=[i[1] for i in self.x],
                    z=[i[2] for i in self.x],
                    mode="markers",
                    name="x_train",
                    marker=dict(
                        size=4,
                        color="aqua",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.y],
                    y=[i[1] for i in self.y],
                    z=[i[2] for i in self.y],
                    mode="markers",
                    name="y_train",
                    marker=dict(
                        size=5,
                        color="mediumspringgreen",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[i[0] for i in x_test],
                    y=[i[1] for i in x_test],
                    z=[i[2] for i in x_test],
                    mode="markers",
                    name="test_x",
                    marker=dict(
                        size=7,
                        color="lightskyblue",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[self.transform(i)[0] for i in x_test],
                    y=[self.transform(i)[1] for i in x_test],
                    z=[self.transform(i)[2] for i in x_test],
                    mode="markers",
                    name="test_x_transsform",
                    marker=dict(
                        size=5,
                        color="black",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
            ]
        )

        if y_test:

            fig.add_trace(
                go.Scatter3d(
                    x=[i[0] for i in y_test],
                    y=[i[1] for i in y_test],
                    z=[i[2] for i in y_test],
                    mode="markers",
                    name="y_test",
                    marker=dict(
                        size=5,
                        color="gold",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                )
            )

        fig.update_layout(scene=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B"))

        fig.write_html(savepath)


# -------------------------------------------------------------------------------------------------------


class test_cc:

    def __init__(self, nullspace_cal, nullspace_eval):

        self.y = nullspace_cal

        self.y_test = nullspace_eval

    def transform(self, c):

        return self.model.transform(c)

    def visualise(self, x_test, path, y_test=False):

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[i[0] for i in self.transformed_x],
                    y=[i[1] for i in self.transformed_x],
                    z=[i[2] for i in self.transformed_x],
                    mode="markers",
                    name="x_train_transformed",
                    marker=dict(
                        size=4,
                        color="magenta",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.x],
                    y=[i[1] for i in self.x],
                    z=[i[2] for i in self.x],
                    mode="markers",
                    name="x_train",
                    marker=dict(
                        size=4,
                        color="aqua",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'circle-x',
                    line=dict(color="MediumPurple", width=1),
                ),
                go.Scatter3d(
                    x=[i[0] for i in self.y],
                    y=[i[1] for i in self.y],
                    z=[i[2] for i in self.y],
                    mode="markers",
                    name="y_train",
                    marker=dict(
                        size=5,
                        color="mediumspringgreen",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[i[0] for i in x_test],
                    y=[i[1] for i in x_test],
                    z=[i[2] for i in x_test],
                    mode="markers",
                    name="test_x",
                    marker=dict(
                        size=7,
                        color="lightskyblue",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
                go.Scatter3d(
                    x=[self.transform(i)[0] for i in x_test],
                    y=[self.transform(i)[1] for i in x_test],
                    z=[self.transform(i)[2] for i in x_test],
                    mode="markers",
                    name="test_x_transsform",
                    marker=dict(
                        size=5,
                        color="black",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                ),
            ]
        )

        if y_test:

            fig.add_trace(
                go.Scatter3d(
                    x=[i[0] for i in y_test],
                    y=[i[1] for i in y_test],
                    z=[i[2] for i in y_test],
                    mode="markers",
                    name="y_test",
                    marker=dict(
                        size=5,
                        color="gold",
                        # colorscale='Viridis',
                        opacity=0.8,
                    ),
                    # symbol = 'star',
                    line=dict(color="DarkSlateGrey", width=2),
                )
            )

        fig.update_layout(scene=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B"))

        fig.write_html(path)

    def fit(self, x, x_test, path, path_vis, x_test_zones):

        self.x = x

        self.x_test = x_test

        params = {
            "iterations": 1250,
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": 0.15,
            "depth": 3,
            "l2_leaf_reg": 1.75,
        }
        self.best_rmse = [-1]

        self.best_mae = [-1]

        self.best_mape = [-1]

        with open(path, "w") as f:

            for iteration_numb in [1500, 2000, 2500]:

                for lr in [0.05, 0.1, 0.15, 0.2]:

                    for depth in [2, 3, 4, 5]:

                        for l2_reg in [0.5, 1, 1.5, 2]:

                            params = {
                                "iterations": iteration_numb,
                                "loss_function": "MultiRMSE",
                                "eval_metric": "MultiRMSE",
                                "learning_rate": lr,
                                "depth": depth,
                                "l2_leaf_reg": l2_reg,
                            }

                            f.write(f"Iterations : {iteration_numb}\n")
                            f.write(f"Learning rate: {lr}\n")
                            f.write(f"Depth : {depth}\n")
                            f.write(f"L2 reg : {l2_reg}\n")

                            f.write("\n")

                            self.model = gbtree_vf(self.y, params, 255)

                            self.model.fit(self.x)

                            self.transformed_x = [
                                self.transform(el_x) for el_x in self.x
                            ]

                            self.transformed_x_test = [
                                self.transform(el_x) for el_x in self.x_test
                            ]

                            rmse_res = np.sqrt(
                                mean_squared_error(self.y, self.transformed_x)
                            )

                            mape_res = mean_absolute_percentage_error(
                                self.y, self.transformed_x
                            )

                            mae_res = mean_absolute_error(self.y, self.transformed_x)

                            f.write("TRAIN: \n")

                            f.write("\n")

                            f.write(f"RMSE : {rmse_res}\n")
                            f.write(f"MAPE : {mape_res}\n")
                            f.write(f"MAE : {mae_res}\n")

                            rmse_res_test = np.sqrt(
                                mean_squared_error(self.y_test, self.transformed_x_test)
                            )

                            mape_res_test = mean_absolute_percentage_error(
                                self.y_test, self.transformed_x_test
                            )

                            mae_res_test = mean_absolute_error(
                                self.y_test, self.transformed_x_test
                            )

                            if (
                                rmse_res_test < self.best_rmse[0]
                                or self.best_rmse[0] == -1
                            ):

                                self.best_rmse = [
                                    rmse_res_test,
                                    iteration_numb,
                                    lr,
                                    depth,
                                    l2_reg,
                                ]

                            if (
                                mae_res_test < self.best_mae[0]
                                or self.best_mae[0] == -1
                            ):

                                self.best_mae = [
                                    mae_res_test,
                                    iteration_numb,
                                    lr,
                                    depth,
                                    l2_reg,
                                ]

                            if (
                                mape_res_test < self.best_mape[0]
                                or self.best_mape[0] == -1
                            ):

                                self.best_mape = [
                                    mape_res_test,
                                    iteration_numb,
                                    lr,
                                    depth,
                                    l2_reg,
                                ]

                            f.write("TEST: \n")

                            f.write("\n")

                            f.write(f"RMSE : {rmse_res_test}\n")
                            f.write(f"MAPE : {mape_res_test}\n")
                            f.write(f"MAE : {mae_res_test}\n")

                            start = time.time()

                            self.model = gbtree_vf(self.y + self.y_test, params, 255)

                            self.model.fit(self.x + self.x_test)

                            f.write("Trained on all: \n")

                            f.write("\n")

                            self.transformed_x = [
                                self.transform(el_x) for el_x in self.x
                            ]

                            self.transformed_x_test = [
                                self.transform(el_x) for el_x in self.x_test
                            ]

                            rmse_res = np.sqrt(
                                mean_squared_error(self.y, self.transformed_x)
                            )

                            mape_res = mean_absolute_percentage_error(
                                self.y, self.transformed_x
                            )

                            mae_res = mean_absolute_error(self.y, self.transformed_x)

                            end = time.time()

                            f.write(f"Time to train: {round(end-start, 3)}\n")

                            f.write("\n")

                            f.write("TRAIN: \n")

                            f.write("\n")

                            f.write(f"RMSE : {rmse_res}\n")
                            f.write(f"MAPE : {mape_res}\n")
                            f.write(f"MAE : {mae_res}\n")

                            rmse_res_test = np.sqrt(
                                mean_squared_error(self.y_test, self.transformed_x_test)
                            )

                            mape_res_test = mean_absolute_percentage_error(
                                self.y_test, self.transformed_x_test
                            )

                            mae_res_test = mean_absolute_error(
                                self.y_test, self.transformed_x_test
                            )

                            f.write("TEST: \n")

                            f.write("\n")

                            f.write(f"RMSE : {rmse_res_test}\n")
                            f.write(f"MAPE : {mape_res_test}\n")
                            f.write(f"MAE : {mae_res_test}\n")

                            f.write("\n")

                            f.write("\n")

                            f.write("\n")

            # print()
            # print(self.best_rmse)

            f.write("Best RMSE: \n")

            f.write("\n")

            f.write(f"RMSE {self.best_rmse[0]}\n")

            f.write(f"Iter {self.best_rmse[1]}\n")

            f.write(f"Lr {self.best_rmse[2]}\n")

            f.write(f"Depth {self.best_rmse[3]}\n")

            f.write(f"L2 {self.best_rmse[4]}\n")

            f.write("\n")

            f.write("\n")

            f.write("Best MAPE: \n")

            f.write("\n")

            f.write(f"MAPE {self.best_mape[0]}\n")

            f.write(f"Iter {self.best_mape[1]}\n")

            f.write(f"Lr {self.best_mape[2]}\n")

            f.write(f"Depth {self.best_mape[3]}\n")

            f.write(f"L2 {self.best_mape[4]}\n")

            f.write("\n")

            f.write("\n")

            f.write("Best MAE: \n")

            f.write("\n")

            f.write(f"MAE {self.best_mae[0]}\n")

            f.write(f"Iter {self.best_mae[1]}\n")

            f.write(f"Lr {self.best_mae[2]}\n")

            f.write(f"Depth {self.best_mae[3]}\n")

            f.write(f"L2 {self.best_mae[4]}\n")

            f.write("\n")

            f.write("\n")

        params = {
            "iterations": self.best_rmse[1],
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": self.best_rmse[2],
            "depth": self.best_rmse[3],
            "l2_leaf_reg": self.best_rmse[4],
        }

        self.model = gbtree_vf(self.y, params, 255)

        self.model.fit(self.x)

        self.transformed_x = [self.transform(el_x) for el_x in self.x]

        self.transformed_x_test = [self.transform(el_x) for el_x in self.x_test]

        self.visualise(x_test_zones, path_vis + "rmse")

        params = {
            "iterations": self.best_mae[1],
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": self.best_mae[2],
            "depth": self.best_mae[3],
            "l2_leaf_reg": self.best_mae[4],
        }

        self.model = gbtree_vf(self.y, params, 255)

        self.model.fit(self.x)

        self.transformed_x = [self.transform(el_x) for el_x in self.x]

        self.transformed_x_test = [self.transform(el_x) for el_x in self.x_test]

        self.visualise(x_test_zones, path_vis + "mae")

        params = {
            "iterations": self.best_mape[1],
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "learning_rate": self.best_mape[2],
            "depth": self.best_mape[3],
            "l2_leaf_reg": self.best_mape[4],
        }

        self.model = gbtree_vf(self.y, params, 255)

        self.model.fit(self.x)

        self.transformed_x = [self.transform(el_x) for el_x in self.x]

        self.transformed_x_test = [self.transform(el_x) for el_x in self.x_test]

        self.visualise(x_test_zones, path_vis + "mape")

        return self.best_mae, self.best_mape, self.best_rmse
