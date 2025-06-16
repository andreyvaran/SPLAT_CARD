import numpy as np
from catboost import CatBoostRegressor
from sklearn.svm import SVR
import cv2

import numpy as np
    
class CCTransformer:
    def __init__(self, 
    source_white, target_white,
    params = {
        'loss_function': 'MultiRMSE',
        #'eval_metric' : 'MultiRMSE',
        'iterations': 10000,
        'learning_rate': 0.2,
        'l2_leaf_reg': 5,
        'random_strength': 1,
        'depth': 16,
        'verbose': 0}, 
    early_stopping_rounds=100, 
    interpolate=False, 
    n_interpolations=1,
    feature_flags = {
        'r': 0, 'g': 0, 'b': 0,
        'rg': 0, 'rb': 0, 'gb': 0,
        'r2':0, 'g2': 0, 'b2': 0,
        'h': 0, 's': 0, 'v': 0,
        'l': 1, 'a': 1, 'b_lab': 1,
        'lms_rel': 0, },
    ):
        
        """
        
        Args:
            params: CatBoost parameters (must include 'loss_function': 'MultiRMSE' cause R^3 => R^3)
            early_stopping_rounds: Early stopping patience for overfitting detection
            interpolate: Whether to add interpolated training data (between each pair of points)
            n_interpolations: Number of points to interpolate between colors (how many points to add between each pair)
        """
        
        assert params.get('loss_function') == 'MultiRMSE', \
               "Must use 'MultiRMSE' loss"
               
        #assert params.get('eval_metric') == 'MultiRMSE', \
               #"Must use 'MultiRMSE' metric"
               # worse result
               
        self.feature_flags = feature_flags
        
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        
        self.interpolate = interpolate
        self.n_interpolations = n_interpolations
        
        self.source_white = source_white
        self.target_white = target_white
        
        self.model = None
        
    def affine_transform_create(self, colors1, colors2):
        
        X = np.column_stack([colors1, np.ones(len(colors1))])
        
        coeffs, _, _, _ = np.linalg.lstsq(X, colors2, rcond=None)
        
        A = coeffs[:3].T
        
        b = coeffs[3]
        
        self.A = A
        
        self.b = b
    
    def affine_transform(self, colors):
        
        original_shape = colors.shape
        
        flattened_colors = np.array(colors).reshape(-1, 3)
        
        transformed = flattened_colors @ self.A.T + self.b
        
        return transformed.reshape(original_shape)
        
    def compute_svr_transform(self, colors1, colors2, kernel='rbf', C=10.0, gamma='scale'):
        
        
        colors1 = np.array(colors1)
        colors2 = np.array(colors2)
        
        models = []
        
        for i in range(3):
            
            model = SVR(kernel=kernel, C=C, gamma=gamma)
            
            model.fit(colors1, colors2[:, i])
            
            models.append(model)
            
        self.svr_models = models

    def svr_transform(self, colors): # works worse
        
        original_shape = colors.shape
        flattened_colors = colors.reshape(-1, 3)
        
        transformed = np.zeros_like(flattened_colors)
        
        for i, model in enumerate(self.svr_models):
            
            transformed[:, i] = model.predict(flattened_colors)
        
        return transformed.reshape(original_shape)

    def augment_lin_interpol(self, colors1, colors2, num_points=1):
        
        if len(colors1) != len(colors2):
            
            raise ValueError("colors1 and colors2 must have same len")
        
        augmented_colors1 = []
        augmented_colors2 = []
        
        for c1, c2 in zip(colors1, colors2):
            
            augmented_colors1.append(c1)
            augmented_colors2.append(c2)
            
            c1 = np.array(c1)
            c2 = np.array(c2)
            
            for alpha in np.linspace(0, 1, num_points + 2)[1:-1]:
                
                interpolated_c1 = c1 * (1 - alpha) + c2 * alpha
                interpolated_c2 = c1 * (1 - alpha) + c2 * alpha  
                
                augmented_colors1.append(interpolated_c1)
                augmented_colors2.append(interpolated_c2)
        
        return np.array(augmented_colors1), np.array(augmented_colors2)
        
    def expand_features(self, x_cal, x_eval, feature_flags):
        """
        Expanding x_cal and x_eval based on feature_flags.
        
        Params:
            x_cal (np.ndarray): Training array shape (n_samples, 3) in RGB [0-1].
            x_eval (np.ndarray): Eval array shape (m_samples, 3) in RGB [0-1].
            feature_flags (dict or list): Dict of flags, showing which features to add.
                                        format: {
                                            'r': bool, 'g': bool, 'b': bool,
                                            'rg': bool, 'rb': bool, 'gb': bool,
                                            'r2': bool, 'g2': bool, 'b2': bool,
                                            'h': bool, 's': bool, 'v': bool,
                                            'l': bool, 'a': bool, 'b_lab': bool,
                                            'lms_rel': bool
                                        }.
        
        returns:
            x_cal_extended (np.ndarray).
            x_eval_extended (np.ndarray).
        """
        flags = feature_flags
        
        # matrix RGB into LMS (for lms_relative)
        bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ])
        
        def _expand_single(x):
            r, g, b = x[0], x[1], x[2]
            new_features = []
            
            # basic channels
            if flags.get('r', False): new_features.append(r)
            if flags.get('g', False): new_features.append(g)
            if flags.get('b', False): new_features.append(b)
            
            # pairwise mult
            if flags.get('rg', False): new_features.append(r * g)
            if flags.get('rb', False): new_features.append(r * b)
            if flags.get('gb', False): new_features.append(g * b)
            
            # squares
            if flags.get('r2', False): new_features.append(r ** 2)
            if flags.get('g2', False): new_features.append(g ** 2)
            if flags.get('b2', False): new_features.append(b ** 2)
            
            # HSV
            if flags.get('h', False) or flags.get('s', False) or flags.get('v', False):
                hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
                if flags.get('h', False): new_features.append(hsv[0])
                if flags.get('s', False): new_features.append(hsv[1])
                if flags.get('v', False): new_features.append(hsv[2])
            
            # Lab
            if flags.get('l', False) or flags.get('a', False) or flags.get('b_lab', False):
                lab = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2Lab)[0][0]
                if flags.get('l', False): new_features.append(lab[0])
                if flags.get('a', False): new_features.append(lab[1])
                if flags.get('b_lab', False): new_features.append(lab[2])
            
            # relative LMS coords
            if flags.get('lms_rel', False):
                lms = bradford_matrix @ np.array([r, g, b])
                lms_white = bradford_matrix @ np.array([1.0, 1.0, 1.0])  # if white = [1, 1, 1]
                new_features.extend(lms / lms_white)
            
            return np.array(new_features)
        
        x_cal_extended = np.array([_expand_single(x) for x in x_cal])
        x_eval_extended = np.array([_expand_single(x) for x in x_eval])
        
        return x_cal_extended, x_eval_extended
        
    def expand_features_for_single_color(self, rgb, feature_flags):

        r, g, b = rgb[0], rgb[1], rgb[2]
        new_features = []


        bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ])

        if feature_flags.get('r', False): new_features.append(r)
        if feature_flags.get('g', False): new_features.append(g)
        if feature_flags.get('b', False): new_features.append(b)

        if feature_flags.get('rg', False): new_features.append(r * g)
        if feature_flags.get('rb', False): new_features.append(r * b)
        if feature_flags.get('gb', False): new_features.append(g * b)

        if feature_flags.get('r2', False): new_features.append(r ** 2)
        if feature_flags.get('g2', False): new_features.append(g ** 2)
        if feature_flags.get('b2', False): new_features.append(b ** 2)

        if any([feature_flags.get(key, False) for key in ['h', 's', 'v']]):
            hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
            if feature_flags.get('h', False): new_features.append(hsv[0])  # Hue (0..180)
            if feature_flags.get('s', False): new_features.append(hsv[1])  # Saturation (0..255)
            if feature_flags.get('v', False): new_features.append(hsv[2])  # Value (0..255)

        if any([feature_flags.get(key, False) for key in ['l', 'a', 'b_lab']]):
            lab = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2Lab)[0][0]
            if feature_flags.get('l', False): new_features.append(lab[0])  # L (0..100)
            if feature_flags.get('a', False): new_features.append(lab[1])  # a (-128..127)
            if feature_flags.get('b_lab', False): new_features.append(lab[2])  # b (-128..127)

        if feature_flags.get('lms_rel', False):
            lms = bradford_matrix @ np.array([r, g, b])
            lms_white = bradford_matrix @ np.array([1.0, 1.0, 1.0])  # white point normalization
            new_features.extend(lms / lms_white)

        return np.array(new_features)

        
    def bradford_adaptation(self, rgb, source_white, target_white):

        bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ])
        
        inv_bradford = np.linalg.inv(bradford_matrix)
        
        lms_source = bradford_matrix @ source_white
        lms_target = bradford_matrix @ target_white
        
        scaling = lms_target / lms_source
        
        lms = bradford_matrix @ rgb
        adapted_lms = lms * scaling
        adapted_rgb = inv_bradford @ adapted_lms
        
        return adapted_rgb
        
    def cat02_adaptation(self, rgb, source_white, target_white):

        cat02_matrix = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834]
        ])
        
        inv_cat02 = np.array([
            [1.0961, -0.2789, 0.1827],
            [0.4544, 0.4735, 0.0721],
            [-0.0096, -0.0057, 1.0153]
        ])
        
        lms_source = cat02_matrix @ source_white
        lms_target = cat02_matrix @ target_white
        
        scaling = lms_target / lms_source
        
        lms = cat02_matrix @ rgb
        adapted_lms = lms * scaling
        adapted_rgb = inv_cat02 @ adapted_lms
        
        return adapted_rgb

    def fit(self, colors_cal_or, nullspace_cal_or, colors_eval_or, nullspace_eval_or, adaptation = 'c'):
        
        """
        Train
        
        Args:
            colors_cal: Training colors [[R,G,B], ...]
            nullspace_cal: Corresponding targets
            
            colors_eval: Validation colors for early stopping
            nullspace_eval: Validation targets
            
        """

        colors_cal = np.array(colors_cal_or)/255.0
        colors_eval = np.array(colors_eval_or)/255.0
        #print(colors_eval*255)

        if adaptation == 'c':
            
            colors_cal = np.array([self.cat02_adaptation(rgb, self.source_white, self.target_white) for rgb in colors_cal])
            colors_eval = np.array([self.cat02_adaptation(rgb, self.source_white, self.target_white) for rgb in colors_eval])
            
        else: 
            
            colors_cal = np.array([self.bradford_adaptation(rgb, self.source_white, self.target_white) for rgb in colors_cal])
            colors_eval = np.array([self.bradford_adaptation(rgb, self.source_white, self.target_white) for rgb in colors_eval])
            
        #print(colors_eval*255)
        
        nullspace_cal = np.array(nullspace_cal_or)/255
        nullspace_eval = np.array(nullspace_eval_or)/255
        
        self.affine_transform_create(colors_cal, nullspace_cal)
        
        colors_cal = self.affine_transform(np.array(colors_cal, dtype=np.float32))
        colors_eval = self.affine_transform(np.array(colors_eval, dtype=np.float32))
        
        #self.compute_svr_transform(colors_cal, nullspace_cal, kernel='rbf', C=2.0, gamma='scale')
        
        #colors_cal = self.svr_transform(colors_cal)
        #colors_eval = self.svr_transform(colors_eval)
        
        if self.interpolate:
        
            colors_cal, nullspace_cal = self.augment_lin_interpol(colors_cal, nullspace_cal, num_points=self.n_interpolations)
            
        X_train, X_val = self.expand_features(colors_cal, colors_eval, self.feature_flags)

        
        #X_train = np.array(colors_cal, dtype=np.float32)
        y_train = np.array(nullspace_cal, dtype=np.float32) - np.array(colors_cal)
        #y_train = np.array(nullspace_cal, dtype=np.float32)
        
        #X_val = np.array(colors_eval, dtype=np.float32)
        y_val = np.array(nullspace_eval, dtype=np.float32) - np.array(colors_eval)
        #y_val = np.array(nullspace_eval, dtype=np.float32)


        # Normalize to [0,1]
        
        #X_train_norm = X_train / 255.0
        #y_train_norm = y_train / 255.0
        
        #X_val_norm = X_val / 255.0
        #y_val_norm = y_val / 255.0

        self.model = CatBoostRegressor(
            **{
                **self.params,
                'early_stopping_rounds': self.early_stopping_rounds,
            }
        )
        
        # Train
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=0
        )

    def transform(self, color_or, printt = False):
        """Applying the trained model to a color"""
        #color = self.svr_transform(color_af)
        
        #if printt:
        
            #print(color_or)
           # print(color_af)
           # print(color)
           
        #print()
        #print()
       # print('originak', color_or)
        
        color_norm = np.array(color_or, dtype=np.float32) / 255.0
        color_br = self.cat02_adaptation(color_norm, self.source_white, self.target_white)
        #print('bradford', list(np.array(color_br)*255))
        color_af = self.affine_transform(np.array(color_br))
        #print('affine', list(np.array(color_af)*255))
        #print(self.feature_flags)
        
        color = self.expand_features_for_single_color(color_af, self.feature_flags)
        
        #print(color)
        
        #print(color)
        
        
        color_pred = np.array(self.model.predict([color])[0]) + np.array(color_af)
        #color_pred = color_af + predicted_diff
        #color_pred = self.bradford_adaptation(color_pred, self.target_white, self.source_white)
        color_pred = np.array(color_pred)*255
        #print('predicted', color_pred)
        #print()
        #print()
        color_pred = np.clip(np.array(color_pred), 0, 255)
        
        # clip in case exceeds 255
        if printt:
            
            return color_pred.astype(float).tolist(), list(np.array(color_af)*255)
            
        else:
            
            return color_pred.astype(float).tolist()
