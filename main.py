"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
# from final.env import ArmEnv
# from final.rl import DDPG

# from _2DOF_Pytorch_test.env import ArmEnv
# # from rl import DDPG
# from _2DOF_Pytorch_test.rl_torch import DDPG

from env import ArmEnv
# from rl import DDPG
from rl_torch import DDPG

MAX_EPISODES = 900
MAX_EP_STEPS = 300
ON_TRAIN = 0 #True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    reward_all = []
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, done, _ = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                reward_all.append(ep_r)
                break
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import os

    plt.figure(figsize=(10, 6))
    plt.ylabel('reward_all')
    plt.xlabel('training steps')
    plt.plot(np.arange(len(reward_all)), reward_all)
    # rl.save()  rl.save()
    plt.show()

    # from datetime import datetime
    #
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = rl.save()
    file_name = f'params_{current_time}.png'  # 文件名
    save_path = './model_save'
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    # print(f"s = {s}")
    env.set_goal(240,240)
    timer = 0
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        print(f"angle_all = {angle_all}")

        # timer +=1
        # if timer % 800 == 200:
        #     env.set_goal(100, 300)
        # if timer % 800 == 400:
        #     env.set_goal(100, 100)
        # if timer % 800 == 600:
        #     env.set_goal(300, 100)
        # if timer % 800 == 0:
        #     env.set_goal(300, 300)


def eval_p2p():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    # s = env.reset()
    s = env.reset_start()
    print(f"s = {s}")
    env.set_goal(240, 240)
    done = 0
    done_4p = 0
    timer = 0
    traj_all = []
    traj_q_all = []

    ang_traj = []
    while not done_4p:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        print(f"s = {s}")
        traj_all.append((s[2]*200,s[3]*200))
        traj_q_all.append((angle_all[0],angle_all[1]))
        print(f"angle_all = {angle_all}")
        timer += 1
        if timer % 800 == 200:
            env.set_goal(240, 240)
            if timer > 800:
                done_4p = 1
        if timer % 800 == 400:
            env.set_goal(220, 220)
        # if timer % 800 == 600:
        #     env.set_goal(100, 100)
        # if timer % 800 == 0:
        #     env.set_goal(300, 100)





    x_vals = [point[0] for point in traj_all]
    y_vals = [point[1] for point in traj_all]

    q1_vals = [point[0] for point in traj_q_all]
    q2_vals = [point[1] for point in traj_q_all]
    print(f"q1_vals = {q1_vals}")
    print(f"q2_vals = {q2_vals}")

    import matplotlib.pyplot as plt

    # 创建三维图形对象
    plt.figure()
    # ax = fig.add_subplot(111, projection='2d')

    # 绘制三维曲线
    # ax.plot(x_vals, y_vals,  label='3D Curve') #z,
    plt.plot(x_vals, y_vals)
    # 设置标签
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    # ax.set_zlabel('Z 轴')

    # # 显示图例
    plt.legend()
    plt.show()


    # 画出关节角度图像
    # 创建三维图形对象
    plt.figure()
    # ax = fig.add_subplot(111, projection='2d')

    # 绘制三维曲线
    # ax.plot(x_vals, y_vals,  label='3D Curve') #z,
    q1_vals = [0 if x > 6.18 else x for x in q1_vals]
    q2_vals = [0 if x > 6.18 else x for x in q2_vals]

    plt.plot(q1_vals, q2_vals)
    # 设置标签
    plt.xlabel('q1_vals ')
    plt.ylabel('q2_vals ')
    # ax.set_zlabel('Z 轴')

    # # 显示图例
    plt.legend()
    plt.show()


if ON_TRAIN:
    train()
else:
    # eval()
    eval_p2p()
    # cde = 0



