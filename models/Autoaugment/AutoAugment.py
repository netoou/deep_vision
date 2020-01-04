from torch.nn import Module
from torch import nn
from torch import optim
from torch.distributions import  Categorical
import torch


from models.Autoaugment.MiniNetTrainer import Trainer
from models.Autoaugment.augmentations import AUGMENT_NAMES

# netoou's autoaugment with custom LSTM controller

class LSTMController(Module):
    def __init__(self, op_space=19, prob_space=11, mag_space=10):
        super(LSTMController, self).__init__()
        self.hidden_units = 100
        self.input_units = 5 * (op_space + 2 * prob_space + 2 * mag_space)
        self.num_layers = 1

        self.lstm = nn.LSTM(self.input_units, self.hidden_units)

        self.op_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_units, 5 * op_space),
        )
        self.prob_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_units, 5 * prob_space * 2),
        )

        self.mag_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_units, 5 * mag_space * 2),
        )

    def forward(self, input):
        out, _ = self.lstm(input)
        # return logits
        return self.op_out(out), self.prob_out(out), self.mag_out(out)


class Critic(Module):
    def __init__(self, op_space=19, prob_space=11, mag_space=10):
        super(Critic, self).__init__()
        # approximate value
        self.critic = nn.Sequential(
            nn.Linear(5 * (op_space + 2 * prob_space + 2 * mag_space), 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, state):
        values = self.critic(state)
        return values


def init_state_zero(op_space=19, prob_space=11, mag_space=10) -> torch.Tensor:
    # current state : previous policy (5 sub-policies which has 2 ops and its prob, mag params)
    # e.g. if we have 19 ops, then the state will have a shape of 1 x (5 * (19 + 11 + 10)) ops, prob, mag
    init_ops = torch.randn((5, (op_space + 2 * prob_space + 2 * mag_space)))
    return init_ops


def logits2policy_argmax(act_op, act_prob, act_mag,
                         op_space=19, prob_space=11, mag_space=10):  # process only single batch
    # TODO Think about duplicate sampling (op1 and op2 might be same,,, there's no unique sampling option)
    op1, op2 = act_op.reshape(-1, op_space).softmax(dim=1).argsort()[:, -2:].split(1, dim=1)
    op1_prob = act_prob.reshape(-1, 2 * prob_space)[:, :prob_space].softmax(dim=1).argmax(dim=1)
    op2_prob = act_prob.reshape(-1, 2 * prob_space)[:, -prob_space:].softmax(dim=1).argmax(dim=1)
    op1_mag = act_mag.reshape(-1, 2 * mag_space)[:, :mag_space].softmax(dim=1).argmax(dim=1)
    op2_mag = act_mag.reshape(-1, 2 * mag_space)[:, -mag_space:].softmax(dim=1).argmax(dim=1)
    policy = list()

    for a, b, c, d, e, f in zip(op1, op1_prob, op1_mag, op2, op2_prob, op2_mag):
        policy.append([(AUGMENT_NAMES[a[0].item()], b.item() / 10, (c.item() + 1) / 10),
                       (AUGMENT_NAMES[d[0].item()], e.item() / 10, (f.item() + 1) / 10)])

    return policy, torch.cat([act_op, act_prob, act_mag], dim=0).reshape(1, -1)


def logits2policy_sampling(act_op, act_prob, act_mag,
                           op_space=19, prob_space=11, mag_space=10):  # process only single batch
    # reshape to (5, -1) : 5 sub policies
    act_ops = act_op.reshape(-1, op_space).softmax(dim=1)
    act_prob1 = act_prob.reshape(-1, 2 * prob_space)[:, :prob_space].softmax(dim=1)
    act_prob2 = act_prob.reshape(-1, 2 * prob_space)[:, -prob_space:].softmax(dim=1)
    act_mag1 = act_mag.reshape(-1, 2 * mag_space)[:, :mag_space].softmax(dim=1)
    act_mag2 = act_mag.reshape(-1, 2 * mag_space)[:, -mag_space:].softmax(dim=1)
    policy = list()
    log_probs = list()

    for sub_op, sub_prob1, sub_prob2, sub_mag1, sub_mag2 in zip(act_ops, act_prob1, act_prob2, act_mag1, act_mag2):
        op_cat = Categorical(sub_op)
        prob1_cat = Categorical(sub_prob1)
        prob2_cat = Categorical(sub_prob2)
        mag1_cat = Categorical(sub_mag1)
        mag2_cat = Categorical(sub_mag2)

        op1, op2 = op_cat.sample((2,))
        prob1 = prob1_cat.sample()
        prob2 = prob2_cat.sample()
        mag1 = mag1_cat.sample()
        mag2 = mag2_cat.sample()

        policy.append([(AUGMENT_NAMES[op1.item()], prob1.item() / 10, (mag1.item() + 1) / 10),
                       (AUGMENT_NAMES[op2.item()], prob2.item() / 10, (mag2.item() + 1) / 10)])
        logprob = op_cat.log_prob(op1) + op_cat.log_prob(op2) + prob1_cat.log_prob(prob1) + prob2_cat.log_prob(prob2)\
                  + mag1_cat.log_prob(mag1) + mag2_cat.log_prob(mag2)
        log_probs.append(logprob.mean())

    # policy for augmentation, policy tensor, distribution's log_prob for backward
    return policy, torch.cat([act_op, act_prob, act_mag], dim=0).reshape(1, -1), torch.stack(log_probs, dim=0).mean()


class AugmentSearch:
    def __init__(self, controller, critic, childnet, datasets, critic_param, controller_param, childnet_param):
        self.controller = controller#(**controller_param)
        self.controller_opt = optim.Adam(self.controller.parameters(), 1e-5)
        self.critic = critic#(**critic_param)
        self.critic_opt = optim.Adam(self.critic.parameters(), 1e-5)

        self.childnet = childnet
        self.childnet_param = childnet_param
        self.datasets = datasets

        self.best_reward = 0.0
        self.best_policy = None

        self.discount_factor = 0.9

        self.op_space = 19
        self.prob_space = 11
        self.mag_space = 10

    def policy_search(self, epochs):
        prev_reward = 0.0
        state = init_state_zero().reshape(1, 1, -1)

        prev_critic = self.critic(state)
        for epoch in range(epochs):
            print(f"Epoch{epoch} start")
            act_op, act_prob, act_mag = self.controller(state.reshape(1, 1, -1))
            print("Search policy")
            policy, new_state, mean_log_prob = logits2policy_sampling(act_op.squeeze(),
                                                                      act_prob.squeeze(),
                                                                      act_mag.squeeze())
            print('Train')
            print(policy)
            trainer = Trainer(self.childnet(**self.childnet_param), self.datasets, 'cpu', 1, 32, 4)
            trainer.set_policy(policy)
            print('Start train')
            top1, top5, loss = trainer.train()  # use top1 as reward
            print(f"End train, top1 : {top1:.4f}, top5 : {top5:.4f}, loss : {loss:.4f}")
            # should be take value from new_state s
            critic_value = self.critic(new_state.detach())
            advantage = top1 - prev_reward - prev_critic.cpu().item() + critic_value.cpu().item()
            prev_reward += advantage * self.discount_factor  # hyperparam (discount factor) at 1.0 position

            critic_loss = advantage * critic_value
            policy_loss = advantage * mean_log_prob

            # update critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # update controller
            self.controller_opt.zero_grad()
            policy_loss.backward()
            self.controller_opt.step()

            state = new_state.detach()
            if top1 > self.best_reward:
                self.best_reward = top1
                self.best_policy = policy

        return self.best_policy, self.best_reward


MAGNITUDES = 10
PROBABILITIES = 11

if __name__=='__main__':
    # lstm = nn.LSTM(5, 100)
    # h_state = torch.randn(1,1,100)
    # c_state = torch.randn(1,1,100)
    # inpts = torch.zeros(10, 1, 5)
    #
    # out, hidden = lstm(inpts)#, (h_state, c_state))
    #
    # print(out.shape)
    # print(hidden[0].shape)
    # print(hidden[1].shape)
    #
    # model = LSTMController()
    # out = model(inpts)
    # print(out[1].shape)
    #
    # print(len(augment_dict))
    # iste = init_state_zero().reshape(1, 1, -1)
    # o, p, m = LSTMController()(iste)
    # print(logits2policy_sampling(o[0], p[0], m[0])[2])
    from datasets.cifar import SmallCifar100
    from models.classification.MobileNetV3 import mobilenet_v3
    from torchvision import transforms
    from torch.utils import data

    device = 'cpu'
    model = mobilenet_v3#(100, 'small').to(device)
    child_param = {
        'n_classes': 100,
        'arc': 'small2',
    }

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    datasets = {
        'train': SmallCifar100('/home/ailab/data/cifar-100-python/', transform=transform, set_type='train'),
        'val': SmallCifar100('/home/ailab/data/cifar-100-python/', transform=transform_val, set_type='test')
    }
    print(len(datasets['train']))
    controller = LSTMController()
    critic = Critic()
    search_obj = AugmentSearch(controller, critic, mobilenet_v3, datasets, None, None, child_param)
    search_obj.policy_search(10)


