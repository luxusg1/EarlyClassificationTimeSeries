from Import import *
from utils import get_padding_sequence,custom_loss_function

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "EarlyClassification.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.x_train, self.x_test, self.y_train, self.y_test = load_gunpoint(return_X_y=True)
        self.brain              = self._build_model()
        self.target_model = self._build_model()
        self.update_number = 0
        self.tau = .125
        self.batch_train = []
        self.target_train_tab = []
        self.memory_0 = deque(maxlen=3000)
        self.memory_1 = deque(maxlen=3000)
        self.memory_2 = deque(maxlen=3000)
        
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        x_train = self.x_train.reshape(self.x_train.shape + (1,1,))
        x_test = self.x_test.reshape(self.x_test.shape + (1,1,))
        x = keras.layers.Input(x_train.shape[1:])
        #    drop_out = Dropout(0.2)(x)
        conv1 = keras.layers.Conv2D(128, kernel_size=8, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        full = keras.layers.GlobalAveragePooling2D()(conv3)
        out = keras.layers.Dense(self.action_size, activation='softmax')(full)


        model = keras.models.Model(inputs=x, outputs=out)

        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss="mse",
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.summary()
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
    
    
    def save_model(self):
        self.brain.save(self.weight_backup)
    
    def target_train(self):
        weights = self.brain.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        
    
    def compute_acc(self):
        count=0
        tab = []
        t = []
        for index in range(len(self.x_train)):
            i = 1
            while(i<=150):
                act = self.brain.predict(np.reshape(get_padding_sequence(self.x_train[index],i),(1,150,1,1)))
                if(np.argmax(act[0])!=0):
                    if np.argmax(act) == self.y_train[index]:
                        count+=1
                    tab.append(np.argmax(act))
                    t.append(i)
                    break 
                if(i==150 and np.argmax(act)==0):
                    if np.argmax(act[0][1:]) + 1 == self.y_train[index]:
                        count+=1
                    tab.append(np.argmax(act[0][1:]) + 1)
                    t.append(i)
                i+=1
        return count/len(self.x_train),tab,t
    
    def compute_acc_val(self):
        count=0
        tab = []
        t = []
        for index in range(len(self.x_test)):
            i = 1
            while(i<=150):
                act = self.brain.predict(np.reshape(get_padding_sequence(self.x_test[index],i),(1,150,1,1)))
                #print(act)
                if(np.argmax(act[0])!=0):
                    if np.argmax(act) == self.y_test[index]:
                        count+=1
                    tab.append(np.argmax(act))
                    t.append(i)
                    break
                if(i==150 and np.argmax(act)==0):
                    if np.argmax(act[0][1:]) + 1 == self.y_test[index]:
                        count+=1
                    tab.append(np.argmax(act[0][1:]) + 1)
                    t.append(i)
                i+=1
        return count/len(self.x_test),tab,t
        
    def save_weight(self):
        self.brain.save_weights("model_last.h5")
        
        
    def act(self, state, time):
        if time < self.state_size:
            if np.random.rand() <= self.exploration_rate:
                return random.randrange(self.action_size)  
            act_values = self.brain.predict(np.reshape(state,(1,self.state_size,1,1)))
            return np.argmax(act_values[0])
        elif time == self.state_size:
            if np.random.rand() <= self.exploration_rate:
                return random.randrange(self.action_size - 1) + 1  
            act_values = self.brain.predict(np.reshape(state,(1,self.state_size,1,1)))
            return np.argmax(act_values[0][1:]) + 1
            
    
    
    def remember(self, state, action, reward, next_state, done):
        if action == 0:  
            self.memory_0.append((state, action, reward, next_state, done))
        if action == 1:
            self.memory_1.append((state, action, reward, next_state, done))
        if action ==2 :
            self.memory_2.append((state, action, reward, next_state, done))
        
        
    def replay(self, sample_batch_size):
        if (len(self.memory_0) < (sample_batch_size/3) or len(self.memory_1) < (sample_batch_size/3) or len(self.memory_2) < (sample_batch_size /3)):
            return
        
        sample_size = int(sample_batch_size/3)
        sample_batch_0 = random.sample(self.memory_0, sample_size)
        sample_batch_1 = random.sample(self.memory_1, sample_size + 1)
        sample_batch_2 = random.sample(self.memory_2, sample_size + 1)
        
        sample_batch_1.extend(sample_batch_2)
        sample_batch_0.extend(sample_batch_1)
        random.shuffle(sample_batch_0)
        sample_batch = sample_batch_0
        self.batch_train = []
        self.target_train_tab = []
        for state, action, reward, next_state, done in sample_batch:
            target = self.target_model.predict(np.reshape(state,(1,self.state_size,1,1)))     
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(np.reshape(next_state,(1,self.state_size,1,1)))[0])
            self.batch_train.append(state)
            self.target_train_tab.append(target)

        train_var = np.reshape(self.batch_train,(sample_batch_size,self.state_size,1,1))
        target_var = np.reshape(self.target_train_tab,(sample_batch_size,3)) 
        self.brain.fit(train_var, target_var, batch_size=32 ,epochs=20, verbose=0)
        self.update_number+=80
        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay