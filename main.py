import tensorflow as tf
import numpy as np
import argparse, os, logging, time
from utils import str2bool, get_entity
from data_loader import read_corpus, tag2label, load_vocabulary, build_vocabulary
from model import BiLSTM_CRF

# Tensorflow settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
# GPU settings
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Hyperparameters
parser = argparse.ArgumentParser(description='NER_MODEL')
parser.add_argument('--train_data', type=str, default='./dataset/train.txt', help='training set')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--hidden_units', type=int, default=300)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--CRF', type=str2bool, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--update_embedding', type=str2bool, default=True)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--shuffle', type=str2bool, default=True)
parser.add_argument('--mode', type=str, default='demo')
parser.add_argument('--model', type=str)
args = parser.parse_args()

# Set Logging
log_path = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.model

output_path = os.path.join('.', "model_save", timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)
log_path['output_path'] = output_path

summary_path = os.path.join(output_path, 'summary')
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
log_path['summary_path'] = summary_path

model_path = os.path.join(output_path, 'checkpoint/')
ckpt_prefix = os.path.join(model_path, 'model')
if not os.path.exists(model_path):
    os.makedirs(model_path)
log_path['model_path'] = ckpt_prefix

result_path = os.path.join(output_path, 'result')
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path['result_path'] = result_path

logger_path = os.path.join(result_path, 'log.txt')
log_path['logger_path'] = logger_path

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
handler = logging.FileHandler(logger_path)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
logging.getLogger().addHandler(handler)

logger.info(str(args))

if os.path.exists('./dataset/vocabulary.pkl'):
    vocab = load_vocabulary('./dataset/vocabulary.pkl')
else:
    build_vocabulary('./dataset/vocabulary.pkl', train_path, 10)
    vocab = load_vocabulary('./dataset/vocabulary.pkl')

# read_dataset & training
if args.mode == 'train':
    train_path = args.train_data
    train_data = read_corpus(train_path)

    model = BiLSTM_CRF(args, tag2label, vocab, log_path, logger, config)
    model.build_graph()
    print('Start training ...')
    print('training data contains : {} lines'.format(len(train_data)))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=model.config) as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess=sess, train=train_data, dev=train_data, saver=saver)

elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    log_path['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, tag2label, vocab, log_path, logger, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('Start demo ...')
        saver.restore(sess, ckpt_file)
        while True:
            print('Please input sentence(pause enter or space to exit):')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('Error for input format, see you next time!')
                break
            else:
                try:
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    IPT = get_entity(tag, demo_sent)
                    print('Key entities: {}'.format(IPT))
                except:
                    print('Please switch to manual service ...')








