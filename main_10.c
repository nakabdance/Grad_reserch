#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Mt.h"

#define W_info_math 0	 //壁情報
#define R_info_math 1	 //右情報
#define T_info_math 2	 //上情報
#define L_info_math 3	 //左情報
#define D_info_math 4	 //下情報
#define G_info_math 5	 //ゴール情報
#define agent_location 6 //エージェントのいる場所

#define MAP_SIZE 5			 //マップの大きさ
#define NUM_GAME 10000		 //ゲーム回数
#define NUM_STEPS 10000		 //エージェントの動ける回数
#define NUM_LEARN 500000		 //学習の回数
#define NUM_CHANGE 10		 //何ステップでゴールを切り替えるか
#define NUM_INPUT 20		 // 入力ノード数。
#define NUM_HIDDEN 3		 // 中間層（隠れ層）の素子数。
#define NUM_CON 3			 //文脈ニューロンの素子数[名嘉]
#define NUM_OUTPUT 2		 // 出力素子数。
#define ALPHA 0.2			 //学習率
#define GAMMA 0.8			 //割引率
#define EPSILON_2 0.05		 // 学習時の重み修正の程度を決める。
#define EPSILON 0.01			 //epsilon greedyに使うepsilon.
#define THRESHOLD_ERROR 0.01 // 学習誤差がこの値以下になるとプログラムは停止する。
#define BETA 0.5			 // 非線形性の強さ
#define NUM_UNPOS_TOP 1		 //教師信号の地域の最大値を表す
#define NUM_UNPOS_BOTTOM 0	 //教師信号の地域の最小値を表す

typedef struct
{
	//現在の座標
	int location_x;
	int location_y;
	int sight_g1[1][1];		 //周囲の視野
	int sight_g2[1][1];		 //方向情報
	int agent_action_select; //移動方向を保持する
	int next_sight_g1[1][1];
	int next_sight_g2[1][1];
	int rewards_goal1_judge; //ゴール１にたどり着いたかどうかの判定に使用する
	int rewards_goal2_judge; //ゴール２にたどり着いたかどうかの判定に使用する
} Agent;

int tx[NUM_STEPS][NUM_INPUT], ty[NUM_STEPS][NUM_OUTPUT], next_tx[NUM_STEPS][NUM_INPUT];				 // 訓練データを格納する配列。tx = 入力値：ty = 教師信号
double x[NUM_INPUT + NUM_CON + 1], h[NUM_HIDDEN + 1], c[NUM_CON], y[NUM_OUTPUT];					 // 閾値表現用に１つ余分に確保。
double next_x[NUM_INPUT + NUM_CON + 1], next_h[NUM_HIDDEN + 1], next_c[NUM_CON], next_y[NUM_OUTPUT]; // 閾値表現用に１つ余分に確保。
double next_maxq;																					 //次状態のQ値の最大値を保持する変数
double w1[NUM_INPUT + NUM_CON + 1][NUM_HIDDEN], w2[NUM_HIDDEN + 1][NUM_OUTPUT];						 // 閾値表現用に１つ余分に確保。
double h_back[NUM_HIDDEN + 1], y_back[NUM_OUTPUT];													 // 隠れ素子、出力素子における逆伝搬量。
int Choice_info;																					 //ポリシーによって決まった情報を保管する変数
int next_info_choice;
int totall_rewards[NUM_LEARN]; //１学習分の報酬を格納する
double qt[4][5];
int Wall_judge;
double *rewards;
int goal1_judge = 0; //ゴール１にたどり着いたかどうかの判定に使用する
int goal2_judge = 0; //ゴール２にたどり着いたかどうかの判定に使用する
int clear_count = 0;
int full_clear_count = 0;

int map_g1[MAP_SIZE][MAP_SIZE] = //ゴール１をめざす
	{
		{5, 3, 3, 3, 1},
		{2, 3, 3, 3, 1},
		{2, 3, 3, 3, 1},
		{2, 3, 3, 3, 1},
		{2, 3, 3, 3, 1}};
int map_g2[MAP_SIZE][MAP_SIZE] = //ゴール２を目指すマップ
	{
		{4, 3, 3, 3, 1},
		{4, 3, 3, 3, 1},
		{4, 3, 3, 3, 1},
		{4, 3, 3, 3, 1},
		{5, 3, 3, 3, 1}};

void showmap(int map_g1[MAP_SIZE][MAP_SIZE], Agent *agent); //マップを可視化する関数
void agent_sight_g1(Agent *agent);
void agent_sight_g2(Agent *agent);
void next_agent_sight_g1(Agent *agent);
void next_agent_sight_g2(Agent *agent);
void agent_action_dc(Agent *agent);																	  //エージェントの行動決定を司る関数
void agent_action_select(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE]);								  //エージェントの状態遷移
double goal_steps(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE], int map_g2[MAP_SIZE][MAP_SIZE], int); //エージェントがゴールに到達したかの判定
void rewards_post(Agent *agent, double c, int, int);
void init_rewards_post(Agent *agent);

void ReadData(Agent *agent, int);
void NextReadData(Agent *agent, int);
void InitNet(void);
void Feedforward(int);
void NextFeedforward(int);
void Backward(int, Agent *agent, int);
void ModifyWaits(void);
double CalcError(int);
void Infoselect(int, Agent *agent, int);
void choice_NextQMAX(Agent *agent);
void init_q_values(double qt[4][5]);
void Act(Agent *agent);
void Act_Learn(Agent *agent, int, int);
void goal_judge(Agent *agent, int);

int main(int argc, char *argv[])
{
	rewards = (double *)malloc(NUM_STEPS * sizeof(double));

	Agent agent;
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	int game;
	int steps;
	int count;
	int ilearn;
	int now_location_x, now_location_y;
	double error, max_error;
	init_genrand(Mtseed);

	for (game = 0; game < NUM_GAME; game++)
	{

		//Agentのスタート位置
		agent.location_x = 2;
		agent.location_y = 2;
		goal1_judge = 0;
		goal2_judge = 0;

		//報酬を入れる配列の初期化
		init_rewards_post(&agent);

		for (steps = 0; steps < NUM_STEPS; steps++)
		{
			count = steps;
			agent_sight_g1(&agent);
			agent_sight_g2(&agent);

			//エージェントが取得した情報をRNN入れるための情報に直して読み込む
			ReadData(&agent, steps);

			//RNNの重みを初期化
			InitNet();

			now_location_x = agent.location_x;
			now_location_y = agent.location_y;
			for (ilearn = 0; ilearn < NUM_LEARN; ilearn++)
			{

				max_error = 0;

				Feedforward(steps);

				Infoselect(steps, &agent, ilearn);

				Act(&agent);

				//行動の選択
				//agent_action_select(&agent, map_g1);

				//選択した行動を実行
				agent_action_dc(&agent);

				//１学習の中でどれだけの報酬を獲得したのかを保持する
				rewards_post(&agent, goal_steps(&agent, map_g1, map_g2, steps), steps, ilearn);

				next_agent_sight_g1(&agent);
				next_agent_sight_g2(&agent);
				NextReadData(&agent, steps);
				NextFeedforward(steps);
				choice_NextQMAX(&agent);

				Backward(steps, &agent, ilearn);
				ModifyWaits();
				Act_Learn(&agent, steps, ilearn);

				if (ilearn < NUM_LEARN)
				{
					agent.location_x = now_location_x;
					agent.location_y = now_location_y;
				}

				error = CalcError(steps);
				if (error > max_error)
				{
					max_error = error;
				}

				if (max_error < THRESHOLD_ERROR)
				{
					break;
				}
			}
			Feedforward(steps);

			Infoselect(steps, &agent, ilearn);

			Act(&agent);

			//行動の選択
			//agent_action_select(&agent, map_g1);

			//選択した行動を実行
			agent_action_dc(&agent);

			//１学習の中でどれだけの報酬を獲得したのかを保持する
			rewards_post(&agent, goal_steps(&agent, map_g1, map_g2, steps), steps, ilearn);
			goal_judge(&agent, steps);

			//printf("%dsteps\n", steps);
			//showmap(map_g1, &agent);

			agent_sight_g1(&agent);
			agent_sight_g2(&agent);

			//printf("エージェントが保持している報酬:%f\n", rewards[steps]);



			if (goal2_judge == 1)
			{
				printf("step: %d\n", (steps+1));
				break;
			}
			if(steps == NUM_STEPS-1)
			{
				printf("学べばか");
				printf("step: %d\n", steps+1);
			}

		}
	}
	printf("full_clear: %d\n", full_clear_count);
	printf("clear     : %d\n", clear_count);
	return (0);
}

void showmap(int map_g1[MAP_SIZE][MAP_SIZE], Agent *agent) //map表示用関数
{

	int mapi = 0;
	int mapn = 0;
	int Smap[MAP_SIZE][MAP_SIZE];
	for (mapi = 0; mapi < MAP_SIZE; mapi++)
	{

		for (mapn = 0; mapn < MAP_SIZE; mapn++)
		{

			Smap[mapi][mapn] = 0;
		}
	}

	for (mapi = 0; mapi < MAP_SIZE; mapi++)
	{

		for (mapn = 0; mapn < MAP_SIZE; mapn++)
		{

			Smap[mapi][mapn] = map_g1[mapi][mapn];
		}
	}

	Smap[(*agent).location_x][(*agent).location_y] = agent_location;

	for (mapi = 0; mapi < MAP_SIZE; mapi++)
	{

		for (mapn = 0; mapn < MAP_SIZE; mapn++)
		{

			if (Smap[mapi][mapn] == W_info_math)
			{

				printf("\x1b[40m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == R_info_math)
			{

				printf("\x1b[41m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == T_info_math)
			{

				printf("\x1b[46m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == L_info_math)
			{

				printf("\x1b[43m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == D_info_math)
			{

				printf("\x1b[44m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == G_info_math)
			{

				printf("\x1b[45m%d.\x1b[m", Smap[mapi][mapn]);
			}
			if (Smap[mapi][mapn] == agent_location)
			{

				printf("\x1b[49m%d.\x1b[m", Smap[mapi][mapn]);
			}
		}

		printf("\n");
	}
}

void agent_sight_g1(Agent *agent)
{

	(*agent).sight_g1[0][0] = 0;

	(*agent).sight_g1[0][0] = map_g1[(*agent).location_x][(*agent).location_y];
}

void agent_sight_g2(Agent *agent)
{

	(*agent).sight_g2[0][0] = 0;
	(*agent).sight_g2[0][0] = map_g2[(*agent).location_x][(*agent).location_y];
}

void next_agent_sight_g1(Agent *agent)
{

	(*agent).next_sight_g1[0][0] = 0;

	(*agent).next_sight_g1[0][0] = map_g1[(*agent).location_x][(*agent).location_y];
}

void next_agent_sight_g2(Agent *agent)
{

	(*agent).next_sight_g2[0][0] = 0;

	(*agent).next_sight_g2[0][0] = map_g2[(*agent).location_x][(*agent).location_y];
}

void agent_action_dc(Agent *agent)
{
	int i = (*agent).agent_action_select;

	switch (i)
	{
	case 0:
		(*agent).location_x = (*agent).location_x - 1;
		if ((*agent).location_x < 0)
		{
			(*agent).location_x = 4;
		}
		break;
	case 1:
		(*agent).location_y = (*agent).location_y + 1;
		if ((*agent).location_y > 4)
		{
			(*agent).location_y = 0;
		}
		break;
	case 2:
		(*agent).location_y = (*agent).location_y - 1;
		if ((*agent).location_y < 0)
		{
			(*agent).location_y = 4;
		}
		break;
	case 3:
		(*agent).location_x = (*agent).location_x + 1;
		if ((*agent).location_x > 4)
		{
			(*agent).location_x = 0;
		}
		break;
	}
}

void init_rewards_post(Agent *agent)
{
	for (int i = 0; i < NUM_STEPS; i++)
	{

		rewards[i] = 0;
	}
}

double goal_steps(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE], int map_g2[MAP_SIZE][MAP_SIZE], int isteps)
{
	double a = 1.0;
	double b = 0.0;

	if (isteps < NUM_CHANGE)
	{
		if (goal1_judge == 0)
		{
			if (map_g1[(*agent).location_x][(*agent).location_y] == G_info_math)
			{
				return a;
			}
			else
			{
				return b;
			}
		}
		else
		{
			if (map_g2[(*agent).location_x][(*agent).location_y] == G_info_math)
			{
				return a;
			}
			else
			{
				return b;
			}
		}
	}
	else
	{
		if (map_g2[(*agent).location_x][(*agent).location_y] == G_info_math)
		{
			return a;
		}
		else
		{
			return b;
		}
	}
}

void rewards_post(Agent *agent, double c, int isteps, int ilearn2)
{
	rewards[isteps] += c;
}

void InitNet(void)
{
	int i, j;

	for (i = 0; i < NUM_INPUT + NUM_CON + 1; i++)
	{
		for (j = 0; j < NUM_HIDDEN; j++)
		{
			w1[i][j] = genrand_real3() - 0.5;
		}
	}

	for (i = 0; i < NUM_HIDDEN + 1; i++)
	{
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			w2[i][j] = genrand_real3() - 0.5;
		}
	}
}

void Feedforward(int isteps)
{
	int i, j;
	double net_input;

	// 順方向の動作
	// 訓練データに従って、ネットワークへの入力を設定する
	for (i = 0; i < NUM_INPUT + NUM_CON; i++)
	{
		if (i < NUM_INPUT)
		{
			x[i] = tx[isteps][i];
		}
		else
		{
			x[i] = c[i - 1];
		}
	}

	// 閾値用に x[NUM_INPUT] = 1.0 とする
	x[NUM_INPUT + NUM_CON] = (double)1.0;

	// 隠れ素子値の計算
	for (j = 0; j < NUM_HIDDEN; j++)
	{
		net_input = 0;
		for (i = 0; i < NUM_INPUT + NUM_CON + 1; i++)
		{
			net_input = net_input + w1[i][j] * x[i];
		}

		h[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));

		// 文脈ニューロン素子値[名嘉]
		c[j] = h[j];
	}
	h[NUM_HIDDEN] = (double)1.0;

	// 出力素子値の計算。
	for (j = 0; j < NUM_OUTPUT; j++)
	{
		net_input = 0;

		for (i = 0; i < NUM_HIDDEN + 1; i++)
		{
			net_input = net_input + w2[i][j] * h[i];
		}
		y[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));
	}
}

void NextFeedforward(int isteps)
{
	int i, j;
	double net_input;

	// 順方向の動作
	// 訓練データに従って、ネットワークへの入力を設定する
	for (i = 0; i < NUM_INPUT + NUM_CON; i++)
	{
		if (i < NUM_INPUT)
		{
			x[i] = next_tx[isteps][i];
		}
		else
		{
			x[i] = c[i - 1];
		}
	}

	// 閾値用に x[NUM_INPUT] = 1.0 とする
	next_x[NUM_INPUT + NUM_CON] = (double)1.0;

	// 隠れ素子値の計算
	for (j = 0; j < NUM_HIDDEN; j++)
	{
		net_input = 0;
		for (i = 0; i < NUM_INPUT + NUM_CON + 1; i++)
		{
			net_input = net_input + w1[i][j] * x[i];
		}

		h[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));

		// 文脈ニューロン素子値[名嘉]
		c[j] = h[j];
	}
	next_h[NUM_HIDDEN] = (double)1.0;

	// 出力素子値の計算。
	for (j = 0; j < NUM_OUTPUT; j++)
	{
		net_input = 0;

		for (i = 0; i < NUM_HIDDEN + 1; i++)
		{
			net_input = net_input + w2[i][j] * h[i];
		}
		y[j] = (double)(1.0 / (1.0 + exp((double)net_input * -BETA)));
	}
}

void Backward(int isteps, Agent *agent, int ilearn2)
{
	int i, j, n;
	double net_input;
	double alpha = ALPHA;
	double gamma = GAMMA;

	for (n = 0; n < NUM_OUTPUT; n++)
	{
		ty[isteps][n] = y[n] + (alpha * (rewards[isteps] + gamma * next_maxq - y[n]));
		if (NUM_UNPOS_TOP < ty[isteps][n]) //教師信号が規定の値域よりも大きかった場合最大規定の値に合わせる
		{
			ty[isteps][n] = 1;
		}
		if (ty[isteps][n] < NUM_UNPOS_BOTTOM) //教師信号が規定の値域よりも小さかった場合最小規定の値に合わせる
		{
			ty[isteps][n] = 0;
		}
	}

	// 逆方向の動作。
	// 出力層素子の逆伝搬時の動作。
	for (j = 0; j < NUM_OUTPUT; j++)
	{
		y_back[j] = BETA * (y[j] - ty[isteps][j]) * ((double)1.0 - y[j]) * y[j];
	}

	// 隠れ層素子の逆伝搬時の動作。
	for (i = 0; i < NUM_HIDDEN; i++)
	{
		net_input = 0;
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			net_input = net_input + w2[i][j] * y_back[j];
		}

		h_back[i] = BETA * net_input * ((double)1.0 - h[i]) * h[i];
	}
}

void ModifyWaits(void)
{
	int i, j;
	double epsilon = (double)EPSILON;

	for (i = 0; i < NUM_INPUT + NUM_CON + 1; i++)
	{
		for (j = 0; j < NUM_HIDDEN; j++)
		{
			w1[i][j] = w1[i][j] - epsilon * x[i] * h_back[j];
		}
	}

	for (i = 0; i < NUM_HIDDEN + 1; i++)
	{
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			w2[i][j] = w2[i][j] - epsilon * h[i] * y_back[j];
		}
	}
}

double CalcError(int isteps)
{
	int i;
	double error = 0.0;

	for (i = 0; i < NUM_OUTPUT; i++)
	{
		error = error + (y[i] - ty[isteps][i]) * (y[i] - ty[isteps][i]);
	}

	error = error / (double)NUM_OUTPUT;

	return (error);
}

void Infoselect(int isteps, Agent *agent, int ilearn2)
{
	double epsilon = EPSILON_2;
	int info_choice = 0;
	int Choice_info = 0;
	int i = 0;
	int j = isteps;
	int n = ilearn2;
	double a = 0.00000001;

	if (genrand_real1() < epsilon)
	{
		i = genrand_int32() % 2;
		if (i == 0)
		{
			info_choice = (*agent).sight_g1[0][0];
		}
		if (i == 1)
		{
			info_choice = (*agent).sight_g2[0][0];
		}
	}
	else
	{
		if (y[1] < y[0])
		{
			info_choice = (*agent).sight_g1[0][0];

			if (j < NUM_CHANGE)
			{
				rewards_post(&agent, a, j, n);
			}
		}
		else
		{
			info_choice = (*agent).sight_g2[0][0];

			if (j > NUM_CHANGE)
			{
				rewards_post(&agent, a, j, n);
			}
		}
	}
	Choice_info = info_choice;
}

void choice_NextQMAX(Agent *agent)
{
	double i;

	if (next_y[1] < next_y[0])
	{
		i = next_y[0];
		next_info_choice = (*agent).next_sight_g1[0][0];
	}
	else
	{
		i = next_y[1];
		next_info_choice = (*agent).next_sight_g2[0][0];
	}
	next_maxq = i;
}

void init_q_values(double qt[4][5])
{
	int i, j;

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 5; j++)
		{
			qt[i][j] = 0;
		}
		qt[i][j] = 0;
	}
}

void Act(Agent *agent)
{
	int i, j, max_q, n, m;
	int num_q[4];
	int same_q[4];
	int same_q_action[4];
	int how_many_same_q;

	double epsilon = EPSILON;
	i = Choice_info - 1;
	max_q = 0;

	for (j = 0; j < 4; j++)
	{
		num_q[j] = qt[j][i];
	}
	if (genrand_real1() < epsilon)
	{
		i = genrand_int32() % 4;
		(*agent).agent_action_select = i;
	}
	else
	{
		max_q = num_q[0];
		for (j = 1; j < 4; j++)
		{
			if (max_q < num_q[j])
			{
				max_q = num_q[j];
				m = j;
			}
		}
		how_many_same_q = 0;
		n = 0;

		for (j = 0; j < 4; j++)
		{
			if (max_q == num_q[j])
			{
				same_q[j] = 1;
				how_many_same_q++;
				same_q_action[n] = j;
				n++;
			}
			else
			{
				same_q[j] = 0;
			}
		}
		if (n != 0)
		{
			(*agent).agent_action_select = same_q_action[genrand_int32() % n];
		}
		else
		{
			(*agent).agent_action_select = m;
		}
	}
}

void Act_Learn(Agent *agent, int isteps, int ilearn2)
{
	int i, j, max_q, n, m;
	int num_q[4];
	int same_q[4];
	int same_q_action[4];
	int how_many_same_q;
	double alpha = ALPHA;
	double gamma = GAMMA;
	double q = qt[(*agent).agent_action_select][Choice_info - 1];

	i = next_info_choice - 1;

	for (j = 0; j < 4; j++)
	{
		num_q[j] = qt[j][i];
	}
	max_q = num_q[0];
	for (j = 1; j < 4; j++)
	{
		if (max_q < num_q[j])
		{
			max_q = num_q[j];
		}
	}
	how_many_same_q = 0;
	n = 0;

	for (j = 0; j < 4; j++)
	{
		if (max_q == num_q[j])
		{
			same_q[j] = 1;
			how_many_same_q++;
			same_q_action[n] = j;
			n++;
		}
		else
		{
			same_q[j] = 0;
		}
	}

	m = same_q_action[genrand_int32() % n];

	qt[(*agent).agent_action_select][Choice_info - 1] = q + (alpha * rewards[isteps] + (gamma * num_q[m]) - q);
}

void goal_judge(Agent *agent, int isteps)
{

	if (isteps < NUM_CHANGE)
	{
		if (goal1_judge == 0)
		{
			if (map_g1[(*agent).location_x][(*agent).location_y] == G_info_math)
			{
				goal1_judge = 1;
				goal2_judge = 0;
			}
		}
		
		else if (goal1_judge == 1)
		{
			if (map_g2[(*agent).location_x][(*agent).location_y] == G_info_math)
			{
			printf("CLEARE");
				goal2_judge = 1;
				full_clear_count++;
			}
		}
		
	}
	else
	{
		if (map_g2[(*agent).location_x][(*agent).location_y] == G_info_math)
		{
			goal2_judge = 1;
			clear_count++;
		}
	}
}

void ReadData(Agent *agent, int isteps)
{
	int sight_g1, sight_g2;
	sight_g1 = (*agent).sight_g1[0][0];
	sight_g2 = (*agent).sight_g2[0][0];

	for (int i = 0; i < NUM_INPUT; i++)
	{
		tx[isteps][i] = 0;
	}
	if ((*agent).location_x == 1)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 1;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 2)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 1;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 3)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 1;
				tx[isteps][13] = 0;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 4)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 1;
				tx[isteps][14] = 0;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 5)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 1;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 1;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 1;
				tx[isteps][18] = 0;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 1;
				tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				tx[isteps][0] = 1;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 1;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 1;
				tx[isteps][3] = 0;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 1;
				tx[isteps][4] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				tx[isteps][0] = 0;
				tx[isteps][1] = 0;
				tx[isteps][2] = 0;
				tx[isteps][3] = 0;
				tx[isteps][4] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				tx[isteps][5] = 1;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 1;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 1;
				tx[isteps][8] = 0;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 1;
				tx[isteps][9] = 0;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				tx[isteps][5] = 0;
				tx[isteps][6] = 0;
				tx[isteps][7] = 0;
				tx[isteps][8] = 0;
				tx[isteps][9] = 1;
				tx[isteps][10] = 0;
				tx[isteps][11] = 0;
				tx[isteps][12] = 0;
				tx[isteps][13] = 0;
				tx[isteps][14] = 1;
				tx[isteps][15] = 0;
				tx[isteps][16] = 0;
				tx[isteps][17] = 0;
				tx[isteps][18] = 0;
				tx[isteps][19] = 1;
			}
		}
	}
}

void NextReadData(Agent *agent, int isteps)
{
	int sight_g1, sight_g2;
	sight_g1 = (*agent).next_sight_g1[0][0];
	sight_g2 = (*agent).next_sight_g2[0][0];

	for (int i = 0; i < NUM_INPUT; i++)
	{
		next_tx[isteps][i] = 0;
	}

	if ((*agent).location_x == 1)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 1;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 2)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 1;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 3)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 1;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 4)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 1;
				next_tx[isteps][14] = 0;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}
		}
	}
	if ((*agent).location_x == 5)
	{
		if ((*agent).location_y == 1)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 1;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 2)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 1;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 3)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 1;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 4)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 1;
				next_tx[isteps][19] = 0;
			}
		}
		if ((*agent).location_y == 5)
		{
			if (sight_g1 == 1)
			{
				next_tx[isteps][0] = 1;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 2)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 1;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 3)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 1;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 4)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 1;
				next_tx[isteps][4] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g1 == 5)
			{
				next_tx[isteps][0] = 0;
				next_tx[isteps][1] = 0;
				next_tx[isteps][2] = 0;
				next_tx[isteps][3] = 0;
				next_tx[isteps][4] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 1)
			{
				next_tx[isteps][5] = 1;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 2)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 1;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 3)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 1;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 4)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 1;
				next_tx[isteps][9] = 0;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}

			if (sight_g2 == 5)
			{
				next_tx[isteps][5] = 0;
				next_tx[isteps][6] = 0;
				next_tx[isteps][7] = 0;
				next_tx[isteps][8] = 0;
				next_tx[isteps][9] = 1;
				next_tx[isteps][10] = 0;
				next_tx[isteps][11] = 0;
				next_tx[isteps][12] = 0;
				next_tx[isteps][13] = 0;
				next_tx[isteps][14] = 1;
				next_tx[isteps][15] = 0;
				next_tx[isteps][16] = 0;
				next_tx[isteps][17] = 0;
				next_tx[isteps][18] = 0;
				next_tx[isteps][19] = 1;
			}
		}
	}
}


