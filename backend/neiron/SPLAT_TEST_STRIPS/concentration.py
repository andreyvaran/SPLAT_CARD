import numpy as np

import plotly.graph_objects as go
import math


def approximate_concentration(analyte_num, c):
    r, g, b = c[0], c[1], c[2]

    if analyte_num == 0:

        x = r - g

        if x < 15:

            return (0.013856821 * x ** 4 + -0.188316172 * x ** 3 +
                    1.0486613 * x ** 2 + 0.136140256 * x + 0.065753348)

        else:

            return 5.4183 * math.exp(x * 0.2761)

    elif analyte_num == 1:

        x = b - r

        if x < -80:

            return 0.0

        elif -80 <= x < 7:

            return (1.48086E-05 * x ** 4 + 0.003563706 * x ** 3 +
                    0.31362197 * x ** 2 + 13.40731133 * x + 283.1927635)

        else:

            return 140.911562 * math.exp(0.133954 * x)

    elif analyte_num == 2:

        x = g - b

        return 0.008457322 * x ** 2 + 0.143561103 * x + 0.695460616

    elif analyte_num == 3:

        x = g - r

        return (2.52582E-06 * x ** 2 + 0.000142481 * x +
                0.014997849 * x + 7.125249973)

    elif analyte_num == 4:

        x = r - g

        if x < -16:

            return 5.0

        else:

            return (-4.70994E-05 * x ** 3 + -0.004102821 * x ** 2 +
                    1.5841948 * x + 49.34061896)

    else:

        return -1

def distance(p1, p2):

    return np.sqrt(np.sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]))


def round_ph(c):

    point_part = c - int(c)

    if point_part > 0 and point_part < 0.25:

        return int(c) + 0.0

    elif point_part >= 0.25 and point_part < 0.75:

        return int(c) + 0.5

    else:

        return int(c) + 1.0


class concentration_approximation:

    def __init__(self, l, cs):

        self.points = l

        self.concentration = cs

    def approximate_R(self, c):

        Rs = [el[0] for el in self.points]

        R = c[0]

        if R <= Rs[0]:

            return [self.concentration[0], self.points[0]]

        if R >= Rs[len(Rs) - 1]:

            return [self.concentration[len(Rs) - 1], self.points[len(Rs) - 1]]

        for i in range(1, len(Rs)):

            if R <= Rs[i] and R >= Rs[i - 1]:

                R_ans = R

                C = (
                    (self.concentration[i] - self.concentration[i - 1])
                    / (Rs[i] - Rs[i - 1])
                ) * (R - Rs[i]) + self.concentration[i]

                if abs(R - Rs[i]) < abs(R - Rs[i - 1]):

                    tmp = [R_ans, self.points[i][1], self.points[i][2]]

                else:

                    tmp = [R_ans, self.points[i - 1][1], self.points[i - 1][2]]

                return [C, tmp]

    def approximate_G(self, c):

        Rs = [el[1] for el in self.points]

        R = c[1]

        if R <= Rs[0]:

            return [self.concentration[0], self.points[0]]

        if R >= Rs[len(Rs) - 1]:

            return [self.concentration[len(Rs) - 1], self.points[len(Rs) - 1]]

        for i in range(1, len(Rs)):

            if R <= Rs[i] and R >= Rs[i - 1]:

                R_ans = R

                C = (
                    (self.concentration[i] - self.concentration[i - 1])
                    / (Rs[i] - Rs[i - 1])
                ) * (R - Rs[i]) + self.concentration[i]

                if abs(R - Rs[i]) < abs(R - Rs[i - 1]):

                    tmp = [R_ans, self.points[i][1], self.points[i][2]]

                else:

                    tmp = [R_ans, self.points[i - 1][1], self.points[i - 1][2]]

                return [C, tmp]

    def approximate_B(self, c):

        Rs = [el[2] for el in self.points]

        R = c[2]

        if R <= Rs[0]:

            return [self.concentration[0], self.points[0]]

        if R >= Rs[len(Rs) - 1]:

            return [self.concentration[len(Rs) - 1], self.points[len(Rs) - 1]]

        for i in range(1, len(Rs)):

            if R <= Rs[i] and R >= Rs[i - 1]:

                R_ans = R

                C = (
                    (self.concentration[i] - self.concentration[i - 1])
                    / (Rs[i] - Rs[i - 1])
                ) * (R - Rs[i]) + self.concentration[i]

                if abs(R - Rs[i]) < abs(R - Rs[i - 1]):

                    tmp = [R_ans, self.points[i][1], self.points[i][2]]

                else:

                    tmp = [R_ans, self.points[i - 1][1], self.points[i - 1][2]]

                return [C, tmp]

    def approximate_1(self, c):

        distances = [[distance(c, self.points[i]), i] for i in range(len(self.points))]

        distances.sort(key=lambda x: x[0])

        p1c = -np.array(self.points[distances[0][1]]) + np.array(c)

        p1p2 = -np.array(self.points[distances[0][1]]) + np.array(
            self.points[distances[1][1]]
        )

        temp = np.array(self.points[distances[0][1]]) + np.array(p1p2) * (
            np.dot(p1c, p1p2) / (np.dot(p1p2, p1p2))
        )

        alpha = (
            self.concentration[distances[0][1]] - self.concentration[distances[1][1]]
        ) / distance(self.points[distances[0][1]], self.points[distances[1][1]])

        if np.dot(p1c, p1p2) > 0:

            alpha *= -1

        c3 = self.concentration[distances[0][1]] + alpha * (
            distance(temp, self.points[distances[0][1]])
        )

        return [c3, temp]

    def approximate_den(self, c):

        coefs = [1.5, -0.2, -0.25]

        # print(c)

        # print('/n/n/n/n/n/n/n/n/n/n/n/n')

        # print(self.points)

        Rs = [
            (coefs[0] * el[0] + coefs[1] * el[1] + coefs[2] * el[2])
            for el in self.points
        ]

        R = coefs[0] * c[0] + coefs[1] * c[1] + coefs[2] * c[2]

        if R <= Rs[0]:

            return [self.concentration[0], self.points[0]]

        if R >= Rs[len(Rs) - 1]:

            return [self.concentration[len(Rs) - 1], self.points[len(Rs) - 1]]

        for i in range(1, len(Rs)):

            if R <= Rs[i] and R >= Rs[i - 1]:

                R_ans = R

                C = (
                    (self.concentration[i] - self.concentration[i - 1])
                    / (Rs[i] - Rs[i - 1])
                ) * (R - Rs[i]) + self.concentration[i]

                if abs(R - Rs[i]) < abs(R - Rs[i - 1]):

                    tmp = [R_ans, self.points[i][1], self.points[i][2]]

                else:

                    tmp = [R_ans, self.points[i - 1][1], self.points[i - 1][2]]

                return [C, tmp]

    def approximate_prot(self, c):

        coefs = [1.1, -0.1, -0.8]

        Rs = [
            (coefs[0] * el[0] + coefs[1] * el[1] + coefs[2] * el[2])
            for el in self.points
        ]

        R = coefs[0] * c[0] + coefs[1] * c[1] + coefs[2] * c[2]

        if R <= Rs[0]:

            return [self.concentration[0], self.points[0]]

        if R >= Rs[len(Rs) - 1]:

            return [self.concentration[len(Rs) - 1], self.points[len(Rs) - 1]]

        for i in range(1, len(Rs)):

            if R <= Rs[i] and R >= Rs[i - 1]:

                R_ans = R

                C = (
                    (self.concentration[i] - self.concentration[i - 1])
                    / (Rs[i] - Rs[i - 1])
                ) * (R - Rs[i]) + self.concentration[i]

                if abs(R - Rs[i]) < abs(R - Rs[i - 1]):

                    tmp = [R_ans, self.points[i][1], self.points[i][2]]

                else:

                    tmp = [R_ans, self.points[i - 1][1], self.points[i - 1][2]]

                return [C, tmp]


def visualise_conc(l, concs, c, ap_c, app_point, savepath):

    l_temp = [c]
    approximated_concentrations = [ap_c]

    approximated_points = [app_point]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[i[0] for i in l_temp],
                y=[i[1] for i in l_temp],
                z=[i[2] for i in l_temp],
                text=[
                    f"{round(approximated_concentrations[el], 2)}"
                    for el in range(len(approximated_concentrations))
                ],
                mode="markers+text",
                marker=dict(
                    size=5,
                    color="magenta",
                    colorscale="Viridis",
                    colorbar=dict(thickness=20),
                    opacity=1,
                ),
                # symbol = 'circle-x',
                line=dict(color="MediumPurple", width=1),
            ),
            go.Scatter3d(
                x=[i[0] for i in approximated_points],
                y=[i[1] for i in approximated_points],
                z=[i[2] for i in approximated_points],
                text=[f"{round(el, 1)}" for el in approximated_concentrations],
                mode="markers+text",
                marker=dict(
                    size=2,
                    color="blue",
                    colorscale="Viridis",
                    colorbar=dict(thickness=20),
                    opacity=1,
                ),
                textfont=dict(color="black", size=6),
            ),
            go.Scatter3d(
                x=[i[0] for i in l],
                y=[i[1] for i in l],
                z=[i[2] for i in l],
                text=[f"{round(el, 1)}" for el in concs],
                mode="markers+text+lines",
                marker=dict(size=7, color="red", opacity=1),
                # symbol = 'circle-x',
                line=dict(color="black", width=2),
            ),
        ]
    )

    for ind in range(len(l_temp)):

        pair_project = [l_temp[ind], approximated_points[ind]]
        fig.add_trace(
            go.Scatter3d(
                x=[i[0] for i in pair_project],
                y=[i[1] for i in pair_project],
                z=[i[2] for i in pair_project],
                mode="lines",
                line=dict(color="green", width=2),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="R",
            yaxis_title="G",
            zaxis_title="B",
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        title=("RGB Calibration points"),
    )

    fig.write_html(savepath)
