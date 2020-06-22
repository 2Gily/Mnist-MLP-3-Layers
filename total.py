###############################################
# 학과 : 컴퓨터공학과               
# 과목 : 인공지능(배유석 교수님)     
# 학번 : 2017150048                
# 이름 : 이길형
# 제출일 : 2020.06.15
# 한국산업기술대학교
################################################ 

# 라이브러리 (import Library)
import numpy as np
import os

# MLP 학습 모델 Python Class로 정의
class MLP:
    # Hyperparameters 초기화
    # 클래스가 실행되면 __init__을 통해 입력된 매개변수를 클래스 내부변수로 초기화
    def __init__(self, hidden_units, minibatch_size, regularization_rate, learning_rate):
        self.hidden_units = hidden_units
        self.minibatch_size = minibatch_size
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate

    # ReLu (Rectified Linear Unit) 함수
    # ReLu를 통해 0보다 작은 값은 0이 반환되고, 0보다 큰 값은 그대로 값이 반환된다.
    # ReLu를 이용해 
    # numpy.zeros(m,n) : m행 n열의 형태의 영행렬을 생성한다. 
    def relu_function(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                ret_vector[i, j] = max(0, matrix_content[i,j])

        return ret_vector

    # the gradient of ReLu (Rectified Linear Unit) 함수
    # ReLu를 통해 1,0 으로 데이터를 이진화한다.
    def grad_relu(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                if matrix_content[i, j] > 0:
                    ret_vector[i, j] = 1
                else:
                    ret_vector[i, j] = 0

        return ret_vector

    # Softmax 함수
    # numpy.exp() : a^x의 형태의 지수함수를 표현한다.
    def softmax_function(self, vector_content):
        return np.exp(vector_content - np.max(vector_content)) / np.sum(np.exp(vector_content - np.max(vector_content)), axis=0)

    # mini-batch 사용, 훈련 데이터에서 무작위를 뽑아 학습
    # yield : return 되는 값을 iterate로 만든다.
    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]  # 만약 input / output shape 체크

        if shuffle: # batch shuffle 적용, 무작위로 설정
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            yield inputs[excerpt], targets[excerpt]

    # MLP 모델을 통한 학습
    # 입력층 : 784개
    # 은닉층 : 입력된 hidden_unit의 개수
    # 출력층 : 10개(0~9)
    def train(self, trainX, trainY, epochs):

        # 매개변수 초기화
        w1_mat = np.random.randn(self.hidden_units, 28*28) *np.sqrt(2./(self.hidden_units+28*28))
        w2_mat = np.random.randn(10, self.hidden_units) *np.sqrt(2./(10+self.hidden_units))
        b1_vec = np.zeros((self.hidden_units, 1))
        b2_vec = np.zeros((10, 1))

        # 입력 / 출력 리사이징 
        # 28*28 형태의 이미지를 1*784의 형태로 정규화
        trainX = np.reshape(trainX, (trainX.shape[0], 28*28)) 
        trainY = np.reshape(trainY, (trainY.shape[0], 1))

        for num_epochs in range(epochs) :
            # if num_epochs % 2 == 0:
            print("Current epoch number : ", num_epochs)

            for batch in self.iterate_minibatches(trainX, trainY, self.minibatch_size, shuffle=True):
                x_batch, y_batch = batch
                x_batch = x_batch.T
                y_batch = y_batch.T

                # Logit 함수의 적용과 ReLu 함수 적용
                z1 = np.dot(w1_mat, x_batch) + b1_vec
                a1 = self.relu_function(z1, self.hidden_units, self.minibatch_size)

                # Logit 함수의 적용과 softmax 함수 적용
                z2 = np.dot(w2_mat, a1) + b2_vec
                a2_softmax = self.softmax_function(z2)

                # 손실함수를 위한 교차 엔트로피
                gt_vector = np.zeros((10, self.minibatch_size))
                for example_num in range(self.minibatch_size):
                    gt_vector[y_batch[0, example_num], example_num] = 1

                # 가중치를 위한 정규화
                d_w2_mat = self.regularization_rate*w2_mat
                d_w1_mat = self.regularization_rate*w1_mat

                # 오차역전파(Back-Propagation)
                delta_2 = np.array(a2_softmax - gt_vector)
                d_w2_mat = d_w2_mat + np.dot(delta_2, (np.matrix(a1)).T)
                d_b2_vec = np.sum(delta_2, axis=1, keepdims=True)

                delta_1 = np.array(np.multiply((np.dot(w2_mat.T, delta_2)), self.grad_relu(z1, self.hidden_units, self.minibatch_size)))
                d_w1_mat = d_w1_mat + np.dot(delta_1, np.matrix(x_batch).T)
                d_b1_vec = np.sum(delta_1, axis=1, keepdims=True)

                d_w2_mat = d_w2_mat/self.minibatch_size
                d_w1_mat = d_w1_mat/self.minibatch_size
                d_b2_vec = d_b2_vec/self.minibatch_size
                d_b1_vec = d_b1_vec/self.minibatch_size

                # 가중치 업데이트
                w2_mat = w2_mat - self.learning_rate*d_w2_mat
                b2_vec = b2_vec - self.learning_rate*d_b2_vec

                w1_mat = w1_mat - self.learning_rate*d_w1_mat
                b1_vec = b1_vec - self.learning_rate*d_b1_vec

        self.w1_mat, self.b1_vec, self.w2_mat, self.b2_vec = w1_mat, b1_vec, w2_mat, b2_vec

    # MLP을 테스트하는 함수
    def test(self, testX):
        output_labels = np.zeros(testX.shape[0])

        num_examples = testX.shape[0]

        testX = np.reshape(testX, (num_examples, 28*28))
        testX = testX.T

        # test with trained model
        z1 = np.dot(self.w1_mat, testX) + self.b1_vec    
        a1 = self.relu_function(z1, self.hidden_units, num_examples)

        z2 = np.dot(self.w2_mat, a1) + self.b2_vec
        a2_softmax = self.softmax_function(z2)

        for i in range(num_examples):
            pred_col = a2_softmax[:, [i]]
            output_labels[i] = np.argmax(pred_col)

        return output_labels

def load_mnist():  # 학습 데이터 60,000 / 테스트 데이터 10,000
    data_dir = './' # 상대경로를 지정

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    # open() : 데이터 파일을 로드한다.
    # numpy.fromfile() : 행렬 데이터를 로드한다.
    # .reshape() : 변수.reshape(전체크기 , 지정할 형태)
    # .astype() : numpy.float 으로 데이터 타입을 변환한다.

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY  # TrainX, TrainY, TestX, TestY

def main():
    trainX, trainY, testX, testY = load_mnist()
    print("Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape)
    
    epochs = 25
    num_hidden_units = 50 
    minibatch_size = 100  
    regularization_rate = 0.01 
    learning_rate = 0.001

    models = MLP(num_hidden_units, minibatch_size, regularization_rate, learning_rate)

    print("MLP 모델을 통한 학습 시작")
    models.train(trainX, trainY, epochs)
    print("MLP 모델을 통한 학습 완료")

    print("학습된 가중치를 이용하여 테스트 시작")
    labels = models.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print("\nTest accuracy: %lf%%" % accuracy)

if __name__ == '__main__':
    main()