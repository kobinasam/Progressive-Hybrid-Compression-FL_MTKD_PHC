import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# === Forced ablation config injected by slurm script ===
_FORCED_ABLATION = "ENTROPY_PRUNE"
# ============================================================
ABLATION_CONFIG = _FORCED_ABLATION   # ENTROPY_PRUNE | ENTROPY_QAT | PRUNE_QAT | REVERSED
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, gzip, io, os
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras.backend as K

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print(f"TensorFlow: {tf.__version__}, GPUs: {len(gpus)}")
print(f"Ablation: {ABLATION_CONFIG}")

# Auto-detect base path
possible_base_paths = [
    Path('/work/hdd/bdcc/msam1/LLaVA-Med/compression'),
    Path('.'),
    Path('/home/maxwellsam/Compression_Model_FL_MTKD'),
]
BASE_PATH = None
for p in possible_base_paths:
    if (p / 'teacher_CNN_model.h5').exists():
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError("teacher_CNN_model.h5 not found")
print(f"Base path: {BASE_PATH}")

IMAGE_SIZE = 200
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 2
NUM_CLIENTS = 3
LEARNING_RATE = 0.001

# Phase rounds (matches main PHC notebook)
PHASE1_ROUNDS = 15
PHASE2_ROUNDS = 25
PHASE3_ROUNDS = 10
TOTAL_ROUNDS = PHASE1_ROUNDS + PHASE2_ROUNDS + PHASE3_ROUNDS

TEMPERATURE = 10.0
ALPHA_KD = 0.6
LAMBDA_ENTROPY = 0.00001
PRUNING_END_SPARSITY = 0.70
LAMBDA_PRUNING = 0.15
QUANTIZATION_BITS = 4
QAT_LEARNING_RATE = 0.0005

# ── Ablation switches ──────────────────────────────────────
ABLATION_FLAGS = {
    "ENTROPY_PRUNE":  {"use_entropy": True,  "use_pruning": True,  "use_qat": False, "reversed": False},
    "ENTROPY_QAT":    {"use_entropy": True,  "use_pruning": False, "use_qat": True,  "reversed": False},
    "PRUNE_QAT":      {"use_entropy": False, "use_pruning": True,  "use_qat": True,  "reversed": False},
    "REVERSED":       {"use_entropy": True,  "use_pruning": True,  "use_qat": True,  "reversed": True},
}
FLAGS = ABLATION_FLAGS[ABLATION_CONFIG]
USE_ENTROPY = FLAGS["use_entropy"]
USE_PRUNING = FLAGS["use_pruning"]
USE_QAT     = FLAGS["use_qat"]
REVERSED    = FLAGS["reversed"]
EFFECTIVE_LAMBDA_ENTROPY = LAMBDA_ENTROPY if USE_ENTROPY else 0.0

print(f"Config: entropy={USE_ENTROPY}, pruning={USE_PRUNING}, qat={USE_QAT}, reversed={REVERSED}")

TEACHER_WEIGHTS = np.array([
    [1.7, 0.85, 0.85],
    [0.6, 1.2, 0.6],
    [0.80, 0.80, 1.6]
], dtype='float32')
TEACHER_WEIGHTS = TEACHER_WEIGHTS / TEACHER_WEIGHTS.sum(axis=1, keepdims=True)

possible_data_paths = [
    Path('/work/hdd/bdcc/msam1/LLaVA-Med/compression/data'),
    Path('./data'),
    Path('/home/maxwellsam/Compression_Model_FL_MTKD/data'),
]
DATA_PATH = None
for p in possible_data_paths:
    if (p / 'Dataset1' / 'Dataset1_input.npy').exists():
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError("Data not found")
print(f"Data path: {DATA_PATH}")

input_data_x_load = np.load(str(DATA_PATH / 'Dataset1' / 'Dataset1_input.npy'))
output_label_y_load = np.load(str(DATA_PATH / 'Dataset1' / 'Dataset1_output.npy'))
input_data_x_2_load = np.load(str(DATA_PATH / 'Dataset2' / 'Dataset2_input.npy'))
output_label_y_2_load = np.load(str(DATA_PATH / 'Dataset2' / 'Dataset2_output.npy'))
input_data_x_3_load = np.load(str(DATA_PATH / 'Dataset3' / 'Dataset3_input.npy'))
output_label_y_3_load = np.load(str(DATA_PATH / 'Dataset3' / 'Dataset3_output.npy'))

train1_x, test1_x, train1_y, test1_y = train_test_split(
    input_data_x_load[2501:3901,:], output_label_y_load[2501:3901,:], test_size=.20, random_state=25)
train2_x, test2_x, train2_y, test2_y = train_test_split(
    input_data_x_2_load[6501:10400,:], output_label_y_2_load[6501:10400,:], test_size=.20, random_state=25)
train3_x, test3_x, train3_y, test3_y = train_test_split(
    input_data_x_3_load[1201:,:], output_label_y_3_load[1201:,:], test_size=.20, random_state=25)

x_test_combined = np.concatenate([test1_x, test2_x, test3_x])
y_test_combined = np.concatenate([test1_y, test2_y, test3_y])
client_data = [(train1_x, train1_y), (train2_x, train2_y), (train3_x, train3_y)]

def create_dataset(data):
    images, labels = data
    return tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
federated_train_data = [create_dataset(d) for d in client_data]
print(f"Test set: {x_test_combined.shape}")

from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers

teacher_CNN = tf.keras.models.load_model(str(BASE_PATH / 'teacher_CNN_model.h5'), compile=False)

_input_dc = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
_rgb_dc = Lambda(lambda x: K.repeat_elements(x, 3, axis=-1))(_input_dc)
_base_nasnet = tf.keras.applications.NASNetLarge(include_top=False, input_tensor=_rgb_dc, weights="imagenet", pooling='avg')
_base_nasnet.trainable = False
_x = BatchNormalization(name="Batch-Normalization-1")(_base_nasnet.output)
_x = Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(_x)
_x = BatchNormalization(name="Batch-Normalization-2")(_x)
_x = Dropout(.2)(_x)
_x = Dense(256, activation='relu')(_x)
_x = BatchNormalization(name="Batch-Normalization-3")(_x)
_x = Dense(NUM_CLASSES, activation="softmax", name="Classifier")(_x)
Deep_COVID_teacher = tf.keras.Model(inputs=_base_nasnet.input, outputs=_x, name="Deep-COVID")
Deep_COVID_teacher.load_weights(str(BASE_PATH / 'Deep_COVID_teacher_model.h5'))

_input_vgg = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
_rgb_vgg = Lambda(lambda x: K.repeat_elements(x, 3, axis=-1))(_input_vgg)
_base_vgg = tf.keras.applications.VGG16(include_top=False, input_tensor=_rgb_vgg, weights='imagenet')
_base_vgg.trainable = False
_x2 = tf.keras.layers.GlobalAveragePooling2D()(_base_vgg.output)
for _u, _d in zip([2756, 1776, 620, 152], [0.65, 0.0, 0.0, 0.65]):
    _x2 = Dense(_u, activation='relu')(_x2); _x2 = Dropout(_d)(_x2)
_x2 = Dense(2, activation='softmax')(_x2)
VGG16_teacher = tf.keras.Model(inputs=_base_vgg.input, outputs=_x2)
VGG16_teacher.load_weights(str(BASE_PATH / 'CNN_Mod_VGG16_teacher_model.h5'))

teachers = [teacher_CNN, Deep_COVID_teacher, VGG16_teacher]
teacher_params = teacher_CNN.count_params()
print(f"Teachers loaded; CNN baseline: {teacher_params:,} params")

student_reference = tf.keras.models.load_model(str(BASE_PATH / 'student_CNN_model.h5'), compile=False)
student_ref_params = student_reference.count_params()
print(f"Student reference: {student_ref_params:,} params")

class EntropyModel(tf.keras.layers.Layer):
    def __init__(self, num_filters=3, **kwargs):
        super().__init__(**kwargs); self.num_filters = num_filters; self._built = False
    def build(self, input_shape):
        if self._built: return
        self.H, self.b, self.a = [], [], []
        filters = [1, self.num_filters, self.num_filters, self.num_filters, 1]
        for i in range(len(filters) - 1):
            init = tf.initializers.RandomUniform(-0.5, 0.5)
            self.H.append(self.add_weight(f'H_{i}', (filters[i+1], filters[i]), initializer=init, trainable=False))
            self.b.append(self.add_weight(f'b_{i}', (filters[i+1], 1), initializer='zeros', trainable=False))
            if i < len(filters) - 2:
                self.a.append(self.add_weight(f'a_{i}', (filters[i+1], 1), initializer='zeros', trainable=False))
        self._built = True; super().build(input_shape)
    def _logits_cumulative(self, x):
        x = tf.reshape(x, (1, -1))
        for i, (H, b) in enumerate(zip(self.H, self.b)):
            x = tf.nn.softplus(H) @ x + b
            if i < len(self.a): x = x + tf.tanh(self.a[i]) * tf.tanh(x)
        return tf.squeeze(x, axis=0)
    def call(self, x, training=False):
        if not self._built: self.build(x.shape)
        x_flat = tf.reshape(x, [-1])
        if training: x_flat = x_flat + tf.random.uniform(tf.shape(x_flat), -0.5, 0.5)
        lower = self._logits_cumulative(x_flat - 0.5)
        upper = self._logits_cumulative(x_flat + 0.5)
        sign = tf.stop_gradient(tf.sign(lower + upper))
        sign = tf.where(tf.equal(sign, 0), tf.ones_like(sign), sign)
        likelihood = tf.maximum(tf.abs(tf.sigmoid(sign * upper) - tf.sigmoid(sign * lower)), 1e-9)
        return tf.reduce_sum(-tf.math.log(likelihood) / tf.math.log(2.0))


class ProgressiveConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = tf.keras.activations.get(activation)
        self.pruning_enabled = False; self.quantization_enabled = False
        self.target_sparsity = 0.0; self.num_bits = 8
    def build(self, input_shape):
        in_c = input_shape[-1]; kh, kw = self.kernel_size
        self.kernel = self.add_weight('kernel', (kh, kw, in_c, self.filters), initializer='glorot_normal', trainable=True)
        self.bias = self.add_weight('bias', (self.filters,), initializer='zeros', trainable=True)
        self.mask_logits = self.add_weight('mask_logits', (kh, kw, in_c, self.filters),
                                           initializer=tf.initializers.Constant(5.0), trainable=False)
        self.entropy_model = EntropyModel(name=f'{self.name}_entropy')
        self.entropy_model(tf.zeros([kh*kw*in_c*self.filters]))
        super().build(input_shape)
    def enable_pruning(self, ts):
        self.pruning_enabled = True; self.target_sparsity = ts
        self.mask_logits.assign(tf.ones_like(self.mask_logits) * 5.0)
    def enable_quantization(self, b): self.quantization_enabled = True; self.num_bits = b
    def get_pruning_mask(self): return tf.nn.sigmoid(self.mask_logits)
    def get_effective_weights(self):
        k = self.kernel
        if self.pruning_enabled: k = k * self.get_pruning_mask()
        return k
    def fake_quantize(self, w):
        wmin, wmax = tf.reduce_min(w), tf.reduce_max(w)
        s = tf.maximum((wmax - wmin) / (2**self.num_bits - 1), 1e-8)
        wq = tf.round((w - wmin) / s) * s + wmin
        return w + tf.stop_gradient(wq - w)
    def compute_pruning_loss(self):
        if not self.pruning_enabled: return 0.0
        return LAMBDA_PRUNING * tf.reduce_mean(self.get_pruning_mask())
    def compute_entropy_loss(self, training=False):
        if EFFECTIVE_LAMBDA_ENTROPY == 0.0: return 0.0
        bits = self.entropy_model(self.get_effective_weights(), training=training)
        return EFFECTIVE_LAMBDA_ENTROPY * bits
    def call(self, inputs, training=False):
        k = self.kernel
        if self.pruning_enabled: k = k * self.get_pruning_mask()
        if self.quantization_enabled and training: k = self.fake_quantize(k)
        out = tf.nn.conv2d(inputs, k, strides=1, padding='VALID')
        out = tf.nn.bias_add(out, self.bias)
        if self.activation: out = self.activation(out)
        return out
    def get_stats(self):
        total = int(tf.size(self.kernel).numpy())
        active = int(tf.reduce_sum(tf.cast(self.get_pruning_mask() > 0.5, tf.float32)).numpy()) if self.pruning_enabled else total
        bits = self.num_bits if self.quantization_enabled else 32
        return {'total_params': total + self.filters, 'active_params': active + self.filters,
                'pruned_ratio': 1.0 - active/total, 'bits': bits,
                'original_bits': (total + self.filters) * 32, 'effective_bits': (active + self.filters) * bits}


class ProgressiveDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs); self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.pruning_enabled = False; self.quantization_enabled = False
        self.target_sparsity = 0.0; self.num_bits = 8
    def build(self, input_shape):
        in_f = input_shape[-1]
        self.kernel = self.add_weight('kernel', (in_f, self.units), initializer='glorot_normal', trainable=True)
        self.bias = self.add_weight('bias', (self.units,), initializer='zeros', trainable=True)
        self.mask_logits = self.add_weight('mask_logits', (in_f, self.units),
                                           initializer=tf.initializers.Constant(5.0), trainable=False)
        self.entropy_model = EntropyModel(name=f'{self.name}_entropy')
        self.entropy_model(tf.zeros([in_f * self.units]))
        super().build(input_shape)
    def enable_pruning(self, ts):
        self.pruning_enabled = True; self.target_sparsity = ts
        self.mask_logits.assign(tf.ones_like(self.mask_logits) * 5.0)
    def enable_quantization(self, b): self.quantization_enabled = True; self.num_bits = b
    def get_pruning_mask(self): return tf.nn.sigmoid(self.mask_logits)
    def get_effective_weights(self):
        k = self.kernel
        if self.pruning_enabled: k = k * self.get_pruning_mask()
        return k
    def fake_quantize(self, w):
        wmin, wmax = tf.reduce_min(w), tf.reduce_max(w)
        s = tf.maximum((wmax - wmin) / (2**self.num_bits - 1), 1e-8)
        wq = tf.round((w - wmin) / s) * s + wmin
        return w + tf.stop_gradient(wq - w)
    def compute_pruning_loss(self):
        if not self.pruning_enabled: return 0.0
        return LAMBDA_PRUNING * tf.reduce_mean(self.get_pruning_mask())
    def compute_entropy_loss(self, training=False):
        if EFFECTIVE_LAMBDA_ENTROPY == 0.0: return 0.0
        bits = self.entropy_model(self.get_effective_weights(), training=training)
        return EFFECTIVE_LAMBDA_ENTROPY * bits
    def call(self, inputs, training=False):
        k = self.kernel
        if self.pruning_enabled: k = k * self.get_pruning_mask()
        if self.quantization_enabled and training: k = self.fake_quantize(k)
        out = tf.matmul(inputs, k) + self.bias
        if self.activation: out = self.activation(out)
        return out
    def get_stats(self):
        total = int(tf.size(self.kernel).numpy())
        active = int(tf.reduce_sum(tf.cast(self.get_pruning_mask() > 0.5, tf.float32)).numpy()) if self.pruning_enabled else total
        bits = self.num_bits if self.quantization_enabled else 32
        return {'total_params': total + self.units, 'active_params': active + self.units,
                'pruned_ratio': 1.0 - active/total, 'bits': bits,
                'original_bits': (total + self.units) * 32, 'effective_bits': (active + self.units) * bits}


def create_progressive_student():
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = ProgressiveConv2D(32, 3, activation='relu', name='prog_conv')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = ProgressiveDense(NUM_CLASSES, name='prog_dense')(x)
    outputs = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs, outputs, name='progressive_student')

print("Layers defined.")

class ProgressiveDistiller:
    def __init__(self, student, teachers, optimizer, alpha=ALPHA_KD, temperature=TEMPERATURE, teacher_weights=None):
        self.student = student; self.teachers = teachers; self.optimizer = optimizer
        self.alpha = alpha; self.temperature = temperature
        self.teacher_weights = teacher_weights or [1.0/len(teachers)]*len(teachers)
        self.ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.kl_loss_fn = tf.keras.losses.KLDivergence()
    def compute_pruning_loss(self):
        total = 0.0
        for layer in self.student.layers:
            if hasattr(layer, 'compute_pruning_loss'): total += layer.compute_pruning_loss()
        return total
    def compute_entropy_loss(self, training=False):
        total = 0.0
        for layer in self.student.layers:
            if hasattr(layer, 'compute_entropy_loss'): total += layer.compute_entropy_loss(training=training)
        return total
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            teacher_preds = [t(x, training=False) for t in self.teachers]
            ensemble = sum(w * p for w, p in zip(self.teacher_weights, teacher_preds))
            soft_s = tf.nn.softmax(tf.math.log(student_pred + 1e-7) / self.temperature)
            soft_t = tf.nn.softmax(tf.math.log(ensemble + 1e-7) / self.temperature)
            ce = self.ce_loss_fn(y, student_pred)
            kd = self.kl_loss_fn(soft_t, soft_s) * (self.temperature ** 2)
            pl = self.compute_pruning_loss()
            el = self.compute_entropy_loss(training=True)
            total = self.alpha * ce + (1 - self.alpha) * kd + pl + el
        grads = tape.gradient(total, self.student.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {'total_loss': total}


def get_model_weights(model): return [w.numpy() for w in model.variables]
def set_model_weights(model, ws):
    for v, w in zip(model.variables, ws): v.assign(w)

def client_update(distiller, dataset, server_weights, num_epochs=NUM_EPOCHS):
    set_model_weights(distiller.student, server_weights)
    losses = []
    for _ in range(num_epochs):
        for batch in dataset:
            x, y = batch
            x = tf.cast(x, tf.float32) / 255.0; y = tf.cast(y, tf.float32)
            if len(x.shape) == 3: x = tf.expand_dims(x, -1)
            r = distiller.training_step(x, y)
            losses.append(r['total_loss'].numpy())
    return {'weights': get_model_weights(distiller.student), 'loss': np.mean(losses)}

def server_aggregate(client_results, current_weights):
    avg = []
    for i in range(len(current_weights)):
        s = np.zeros_like(current_weights[i])
        for r in client_results: s += r['weights'][i]
        avg.append(s / len(client_results))
    return avg

def evaluate(model, x_test, y_test):
    x = x_test.astype(np.float32) / 255.0
    if len(x.shape) == 3: x = np.expand_dims(x, -1)
    y = y_test.astype(np.float32)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model.evaluate(x, y, verbose=0)


def estimate_entropy_size(model):
    total_bits = 0; total_params = 0
    for layer in model.layers:
        if hasattr(layer, 'entropy_model') and hasattr(layer, 'get_effective_weights'):
            eff = layer.get_effective_weights()
            bits = layer.entropy_model(eff, training=False).numpy()
            total_bits += bits
            total_params += int(tf.size(layer.kernel).numpy())
    original_bits = student_ref_params * 32
    return {'cr': original_bits / max(total_bits, 1), 'bpp': total_bits / max(total_params, 1),
            'compressed_kb': total_bits / 8 / 1024}

def estimate_gzip(model):
    arrays = []
    for layer in model.layers:
        if hasattr(layer, 'get_effective_weights'):
            arrays.append(layer.get_effective_weights().numpy())
            arrays.append(layer.bias.numpy())
    buf = io.BytesIO()
    for a in arrays: buf.write(a.astype(np.float32).tobytes())
    raw = buf.getvalue()
    comp = gzip.compress(raw, compresslevel=9)
    return {'cr': len(raw) / max(len(comp), 1), 'compressed_kb': len(comp) / 1024,
            'bw_red': (1 - len(comp)/len(raw)) * 100}


def transition_to_pruning(model):
    print("\n>>> Enabling pruning")
    for layer in model.layers:
        if isinstance(layer, (ProgressiveConv2D, ProgressiveDense)):
            layer.enable_pruning(PRUNING_END_SPARSITY)
            layer.mask_logits._trainable = True

def update_pruning_sparsity(model, current_round, p_start, p_end):
    progress = min(max((current_round - p_start) / (p_end - p_start), 0.0), 1.0)
    target = (progress ** 2) * PRUNING_END_SPARSITY
    for layer in model.layers:
        if isinstance(layer, (ProgressiveConv2D, ProgressiveDense)):
            layer.target_sparsity = target
            mag = tf.abs(layer.kernel)
            flat_mag = tf.reshape(mag, [-1])
            n = int(tf.size(flat_mag).numpy())
            n_prune = int(target * n)
            if n_prune > 0:
                threshold = tf.sort(flat_mag)[n_prune - 1]
                flat_logits = tf.reshape(layer.mask_logits, [-1])
                nudge = tf.where(tf.reshape(mag, [-1]) <= threshold, flat_logits - 0.5, flat_logits + 0.2)
                layer.mask_logits.assign(tf.reshape(nudge, layer.mask_logits.shape))

def transition_to_qat(model):
    print("\n>>> Enabling QAT")
    for layer in model.layers:
        if isinstance(layer, (ProgressiveConv2D, ProgressiveDense)):
            if layer.pruning_enabled:
                mask = layer.get_pruning_mask()
                layer.mask_logits.assign(tf.where(mask > 0.5, 10.0, -10.0))
                layer.mask_logits._trainable = False
            layer.enable_quantization(QUANTIZATION_BITS)

print("Utilities defined.")

global_model = create_progressive_student()
dummy_input = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1))
_ = global_model(dummy_input)
ref_weights = student_reference.get_weights()
prog_conv = global_model.get_layer('prog_conv')
prog_dense = global_model.get_layer('prog_dense')
prog_conv.kernel.assign(ref_weights[0]); prog_conv.bias.assign(ref_weights[1])
prog_dense.kernel.assign(ref_weights[2]); prog_dense.bias.assign(ref_weights[3])
global_weights = get_model_weights(global_model)
init_loss, init_acc = evaluate(global_model, x_test_combined, y_test_combined)
print(f"Initial accuracy: {init_acc*100:.2f}%")

def get_phase_schedule():
    """
    Returns list of (round_start, round_end, phase_name, transition_fn_or_none).
    For ablations, skipped techniques map to 'pretrain' (still train with KD only).
    """
    schedule = []
    if REVERSED:
        # QAT first, then prune, then entropy-only KD
        schedule = [
            (1,                  PHASE3_ROUNDS,        'QAT',     'qat'),
            (PHASE3_ROUNDS+1,    PHASE3_ROUNDS+PHASE2_ROUNDS, 'PRUNE', 'prune'),
            (PHASE3_ROUNDS+PHASE2_ROUNDS+1, TOTAL_ROUNDS, 'ENTROPY_ONLY', None),
        ]
    else:
        # Normal order, but skip phases that are disabled
        cur = 1
        # Phase 1 always runs (KD + maybe entropy)
        schedule.append((cur, cur + PHASE1_ROUNDS - 1, 'KD+ENT' if USE_ENTROPY else 'KD_ONLY', None))
        cur += PHASE1_ROUNDS
        # Phase 2: pruning if enabled, else extended KD
        if USE_PRUNING:
            schedule.append((cur, cur + PHASE2_ROUNDS - 1, 'KD+PRUNE', 'prune'))
        else:
            schedule.append((cur, cur + PHASE2_ROUNDS - 1, 'KD_EXTENDED', None))
        cur += PHASE2_ROUNDS
        # Phase 3: QAT if enabled, else extended KD
        if USE_QAT:
            schedule.append((cur, cur + PHASE3_ROUNDS - 1, 'KD+QAT', 'qat'))
        else:
            schedule.append((cur, cur + PHASE3_ROUNDS - 1, 'KD_FINAL', None))
        cur += PHASE3_ROUNDS
    return schedule

SCHEDULE = get_phase_schedule()
print("Phase schedule:")
for s, e, name, tr in SCHEDULE:
    print(f"  Rounds {s:3d}--{e:3d}  {name:<15}  transition={tr}")

history = {'round': [], 'loss': [], 'accuracy': [], 'cr_ent': [], 'cr_gzip': [],
           'bpp': [], 'phase': [], 'sparsity': []}

print("\n" + "="*70)
print(f"ABLATION TRAINING: {ABLATION_CONFIG}")
print("="*70)

best_acc_per_phase = {}
best_weights_per_phase = {}
current_phase_idx = 0

for round_num in range(TOTAL_ROUNDS):
    cur_round = round_num + 1
    
    # Find current phase
    for idx, (s, e, name, tr) in enumerate(SCHEDULE):
        if s <= cur_round <= e:
            phase_idx = idx
            phase_name = name
            transition = tr
            phase_start = s
            phase_end = e
            break
    
    # Trigger transition at start of phase
    if cur_round == phase_start and phase_idx != current_phase_idx:
        # Restore best from previous phase
        prev_phase_idx = current_phase_idx
        if prev_phase_idx in best_weights_per_phase:
            set_model_weights(global_model, best_weights_per_phase[prev_phase_idx])
            print(f"  [Restored best from phase {prev_phase_idx}: {best_acc_per_phase[prev_phase_idx]*100:.2f}%]")
        if transition == 'prune':
            transition_to_pruning(global_model)
        elif transition == 'qat':
            transition_to_qat(global_model)
        global_weights = get_model_weights(global_model)
        current_phase_idx = phase_idx
    
    # Pruning sparsity update
    if phase_name in ('KD+PRUNE', 'PRUNE'):
        update_pruning_sparsity(global_model, cur_round, phase_start, phase_end)
        global_weights = get_model_weights(global_model)
    
    # LR
    lr = QAT_LEARNING_RATE if 'QAT' in phase_name else LEARNING_RATE
    
    # Federated round
    client_results = []
    for client_id, client_dataset in enumerate(federated_train_data):
        client_model = create_progressive_student()
        _ = client_model(dummy_input)
        for g_l, c_l in zip(global_model.layers, client_model.layers):
            if isinstance(g_l, (ProgressiveConv2D, ProgressiveDense)):
                c_l.pruning_enabled = g_l.pruning_enabled
                c_l.quantization_enabled = g_l.quantization_enabled
                c_l.target_sparsity = g_l.target_sparsity
                c_l.num_bits = g_l.num_bits
                if g_l.pruning_enabled:
                    c_l.mask_logits._trainable = g_l.mask_logits._trainable
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        distiller = ProgressiveDistiller(client_model, teachers, opt, teacher_weights=TEACHER_WEIGHTS[client_id].tolist())
        result = client_update(distiller, client_dataset, global_weights)
        client_results.append(result)
    
    global_weights = server_aggregate(client_results, global_weights)
    set_model_weights(global_model, global_weights)
    
    loss, accuracy = evaluate(global_model, x_test_combined, y_test_combined)
    ent = estimate_entropy_size(global_model)
    gz = estimate_gzip(global_model)
    sparsity = max((l.get_stats()['pruned_ratio'] for l in global_model.layers if hasattr(l, 'get_stats')), default=0.0)
    
    if accuracy > best_acc_per_phase.get(current_phase_idx, 0.0):
        best_acc_per_phase[current_phase_idx] = accuracy
        best_weights_per_phase[current_phase_idx] = [w.copy() for w in global_weights]
    
    history['round'].append(cur_round)
    history['loss'].append(loss)
    history['accuracy'].append(accuracy)
    history['cr_ent'].append(ent['cr'])
    history['cr_gzip'].append(gz['cr'])
    history['bpp'].append(ent['bpp'])
    history['phase'].append(phase_name)
    history['sparsity'].append(sparsity)
    
    if cur_round % 5 == 0 or cur_round == 1:
        print(f"R{cur_round:3d} [{phase_name:<15}] Acc:{accuracy*100:6.2f}% | CR(ent):{ent['cr']:6.1f}x | CR(gzip):{gz['cr']:5.2f}x | bpp:{ent['bpp']:6.3f} | spars:{sparsity*100:4.1f}%")

print("="*70); print("Training complete!")

final_loss, final_acc = evaluate(global_model, x_test_combined, y_test_combined)
final_ent = estimate_entropy_size(global_model)
final_gz = estimate_gzip(global_model)

print("\n" + "="*70)
print(f"FINAL RESULTS - {ABLATION_CONFIG}")
print("="*70)
print(f"Accuracy:        {final_acc*100:.2f}%")
print(f"Loss:            {final_loss:.4f}")
print(f"CR (entropy):    {final_ent['cr']:.2f}x")
print(f"CR (gzip):       {final_gz['cr']:.2f}x")
print(f"Bits/param:      {final_ent['bpp']:.4f}")
print(f"BW reduction:    {final_gz['bw_red']:.1f}%")
print(f"Compressed size: {final_gz['compressed_kb']:.2f} KB")

results = {
    'ablation_config': ABLATION_CONFIG,
    'flags': FLAGS,
    'final_accuracy': float(final_acc),
    'final_loss': float(final_loss),
    'cr_entropy': float(final_ent['cr']),
    'cr_gzip': float(final_gz['cr']),
    'bits_per_param': float(final_ent['bpp']),
    'bandwidth_reduction': float(final_gz['bw_red']),
    'compressed_size_kb': float(final_gz['compressed_kb']),
    'history': {k: [float(v) if isinstance(v, (float, np.floating)) else (str(v) if isinstance(v, str) else v) for v in vals]
                for k, vals in history.items()},
}

out_json = f'ablation_{ABLATION_CONFIG}_results.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")
