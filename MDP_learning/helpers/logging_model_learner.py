import os.path
from time import strftime, gmtime
import inspect
from keras.callbacks import TensorBoard
from keras.models import model_from_yaml


class LoggingModelLearner(object):
    def __init__(self, environment, sequence_length,
                 write_tboard=True,
                 out_dir_add=None):
        # some trickery to get the child class filename
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        base_name = os.path.splitext(os.path.basename(module.__file__))[0]

        self.env = environment
        self.models = []
        self.useRNN = sequence_length > 0
        self.sequence_length = sequence_length

        self.out_dir = './out/{}/{}_{}_seqlen{}{}'.format(
            base_name,
            self.env.spec.id if self.env.spec is not None else str(self.env),
            strftime("%y-%m-%d_%H:%M", gmtime()),
            self.sequence_length,
            '__{}'.format(out_dir_add) if out_dir_add is not None else ''
        )

        self.Ttensorboard = [TensorBoard(log_dir='{}/logs/Tlearn'.format(self.out_dir))] if write_tboard else []
        self.Rtensorboard = [TensorBoard(log_dir='{}/logs/Rlearn'.format(self.out_dir))] if write_tboard else []
        self.Dtensorboard = [TensorBoard(log_dir='{}/logs/Dlearn'.format(self.out_dir))] if write_tboard else []

    def save_model_config(self):
        for i, m in enumerate(self.models):
            serialize_model(m, "{}/model{}".format(self.out_dir, i))

    def save(self):
        self.save_model_config()
        for i, m in enumerate(self.models):
            serialize_weights(m, "{}/model{}".format(self.out_dir, i))

    def load(self, n_models=3):
        self.models = []
        for i in range(n_models):
            self.models.append(deserialize_model("{}/model{}".format(self.out_dir, i)))
        if hasattr(self, 'tmodel'):
            self.tmodel = self.models[0]
        if hasattr(self, 'rmodel'):
            self.rmodel = self.models[1]
        if hasattr(self, 'dmodel'):
            self.dmodel = self.models[2]


def serialize_model(model, folder, filename='model'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("{}/{}.yaml".format(folder, filename), "w") as yaml_file:
        yaml_file.write(model_yaml)
    print("Saved {} to {}".format(folder, filename))


def serialize_weights(model, folder, filename='weights'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # serialize weights to HDF5
    model.save_weights("{}/{}.h5".format(folder, filename))
    print("Saved {} to {}".format(folder, filename))


def deserialize_model(folder, filenames=('model', 'weights')):
    # load YAML and create model
    yaml_file = open("{}/{}.yaml".format(folder, filenames[0]), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("{}/{}.h5".format(folder, filenames[1]))
    print("Loaded model from {}".format(folder))
    return loaded_model_yaml
