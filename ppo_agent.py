import numpy as np
import torch as T
from memory import PPOMemory
from models import ActorNetwork, CriticNetwork
# from models import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, depth_inp_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, ent_coef = 0.01, batch_size=64, n_epochs=10, load_models = False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef

        self.actor = ActorNetwork(n_actions, depth_inp_dims, alpha)
        self.critic = CriticNetwork(depth_inp_dims, alpha)
        if load_models:
            self.load_models()
        self.memory = PPOMemory(batch_size)
       
    def remember(self, depth_map, angle, action, probs, vals, reward, done):
        self.memory.store_memory(depth_map, angle, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, depth_map, angle):
        #state = T.tensor([state], dtype=T.float).to(self.actor.device)
        depth_map = T.tensor(np.expand_dims(depth_map,0), dtype=T.float32)
        angle = T.tensor(np.expand_dims(angle, 0), dtype=T.float32)
        
        dist = self.actor(depth_map, angle)
        value = self.critic(depth_map, angle)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for epoch in range(self.n_epochs):
            depth_map_arr, angle_arr, action_arr, old_prob_arr, vals_arr, \
                  reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            loss_hist = []
            for batch in batches:
                #states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                depth_map = T.tensor(depth_map_arr[batch], dtype=T.float32)
                angle = T.tensor(angle_arr[batch], dtype = T.float32)
                
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(depth_map, angle)
                critic_value = self.critic(depth_map, angle)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                # entropy loss
                entropy = dist.entropy()
                entropy_loss = -self.ent_coef*entropy.mean()

                total_loss = actor_loss + 0.5*critic_loss + entropy_loss
                # total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                #loss_hist.append(total_loss.detach().numpy())
            #print(f'epoch : {epoch}, loss : {np.array(loss_hist)}') # for monitoring

        self.memory.clear_memory() 
