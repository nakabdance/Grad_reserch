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

#define MAP_SIZE 11			 //マップの大きさ
#define NUM_GAME 1			 //ゲーム回数
#define NUM_STEPS 500		 //エージェントの動ける回数
#define NUM_LEARN 5000		 //学習の回数
#define NUM_CHANGE 250		 //何ステップでゴールを切り替えるか
#define NUM_SAMPLE 6561		 // 訓練データのサンプル数。
#define NUM_INPUT 10		 // 入力ノード数。
#define NUM_HIDDEN 3		 // 中間層（隠れ層）の素子数。
#define NUM_CON 3			 //文脈ニューロンの素子数[名嘉]
#define NUM_OUTPUT 2		 // 出力素子数。
#define ALPHA 0.1			 //学習率
#define GAMMA 0.90			 //割引率
#define EPSILON_2 0.1		 // 学習時の重み修正の程度を決める。
#define EPSILON 0.3			 //epsilon greedyに使うepsilon.
#define THRESHOLD_ERROR 0.01 // 学習誤差がこの値以下になるとプログラムは停止する。
#define BETA 0.5			 // 非線形性の強さ

typedef struct
{
	//現在の座標
	int location_x;
	int location_y;
	int sight_g1[1][1];		 //周囲の視野
	int sight_g2[1][1];		 //方向情報
	int agent_action_select; //移動方向を保持する
	int step_count;			 //全学習のステップ数を保持する
	double rewards[NUM_STEPS];
	int next_sight_g1[1][1];
	int next_sight_g2[1][1];

} Agent;

int tx[NUM_SAMPLE][NUM_INPUT], ty[NUM_SAMPLE][NUM_OUTPUT], next_tx[NUM_SAMPLE][NUM_INPUT];			 // 訓練データを格納する配列。tx = 入力値：ty = 教師信号
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

int map_g1[MAP_SIZE][MAP_SIZE] = //ゴール１をめざす
	{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 5, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
int map_g2[MAP_SIZE][MAP_SIZE] = //ゴール２を目指すマップ
	{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 5, 3, 3, 3, 3, 3, 3, 3, 3, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

void showmap(int map_g1[MAP_SIZE][MAP_SIZE], Agent *agent); //マップを可視化する関数
//void make_Smap(int map_g1[MAP_SIZE][MAP_SIZE]);
void agent_sight_g1(Agent *agent);
void agent_sight_g2(Agent *agent);
void next_agent_sight_g1(Agent *agent);
void next_agent_sight_g2(Agent *agent);
void agent_action_dc(Agent *agent);															  //エージェントの行動決定を司る関数
void agent_action_select(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE]);						  //エージェントの状態遷移
int goal_steps(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE], int map_g2[MAP_SIZE][MAP_SIZE]); //エージェントがゴールに到達したかの判定
void rewards_post(Agent *agent, int c, int);
void init_rewards_post(Agent *agent);

void ReadData(Agent *agent, int);
void NextReadData(Agent *agent, int);
void InitNet(void);
void Feedforward(int);
void NextFeedforward(int);
void Backward(int, Agent *agent);
void ModifyWaits(void);
void Infoselect(int, Agent *agent);
void choice_NextQMAX(Agent *agent);
void init_q_values(double qt[4][5]);
void Act(Agent *agent);

int main(int argc, char *argv[])
{

	Agent agent;
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	int game;
	int steps;
	int count;
	init_genrand(Mtseed);
	agent.step_count = 0;

	for (game = 0; game < NUM_GAME; game++)
	{

		//Agentのスタート位置
		agent.location_x = 5;
		agent.location_y = 10;

		init_rewards_post(&agent);

		for (steps = 0; steps < NUM_STEPS; steps++)
		{
			count = steps;
			printf("before info :\n");
			agent_sight_g1(&agent);
			agent_sight_g2(&agent);

			/*エージェントが取得した情報をRNN入れるための情報に直して読み込む*/
			ReadData(&agent, steps);
			printf("\n");

			printf("エージェントが保持している報酬:%f\n", agent.rewards[steps]);

			Feedforward(steps);
			Infoselect(steps, &agent);

			printf("実際に使う情報%d\n", Choice_info);

			printf("%d回目の行動", count + 1);

			printf("\n");

			Act(&agent);
			/*行動の選択*/
			agent_action_select(&agent, map_g1);
			/*選択した行動を実行*/
			agent_action_dc(&agent);
			/*１学習の中でどれだけの報酬を獲得したのかを保持する*/
			rewards_post(&agent, goal_steps(&agent, map_g1, map_g2), steps);

			showmap(map_g1, &agent);

			printf("after info :\n");
			agent_sight_g1(&agent);
			agent_sight_g2(&agent);
			next_agent_sight_g1(&agent);
			next_agent_sight_g2(&agent);
			printf("エージェントが保持している報酬:%f\n", agent.rewards[steps]);

			printf("\n");

			NextReadData(&agent, steps);
			NextFeedforward(steps);
			choice_NextQMAX(&agent);
			Backward(steps, &agent);
			ModifyWaits();
		}
	}

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

	//(*agent).sight[0][0] = map_g1[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).sight[0][1] = map_g1[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).sight[0][2] = map_g1[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).sight[1][0] = map_g1[(*agent).location_x][(*agent).location_y - 1];
	(*agent).sight_g1[0][0] = map_g1[(*agent).location_x][(*agent).location_y];
	//(*agent).sight[1][2] = map_g1[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).sight[2][0] = map_g1[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).sight[2][1] = map_g1[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).sight[2][2] = map_g1[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("to g1 info > %d\n", (*agent).sight_g1[0][0]);
}

void agent_sight_g2(Agent *agent)
{

	(*agent).sight_g2[0][0] = 0;

	//(*agent).sight_g2[0][0] = map_g2[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).sight_g2[0][1] = map_g2[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).sight_g2[0][2] = map_g2[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).sight_g2[1][0] = map_g2[(*agent).location_x][(*agent).location_y - 1];
	(*agent).sight_g2[0][0] = map_g2[(*agent).location_x][(*agent).location_y];
	//(*agent).sight_g2[1][2] = map_g2[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).sight_g2[2][0] = map_g2[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).sight_g2[2][1] = map_g2[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).sight_g2[2][2] = map_g2[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("to g2 info > %d\n", (*agent).sight_g2[0][0]);
}

void next_agent_sight_g1(Agent *agent)
{

	(*agent).next_sight_g1[0][0] = 0;

	//(*agent).sight[0][0] = map_g1[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).sight[0][1] = map_g1[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).sight[0][2] = map_g1[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).sight[1][0] = map_g1[(*agent).location_x][(*agent).location_y - 1];
	(*agent).next_sight_g1[0][0] = map_g1[(*agent).location_x][(*agent).location_y];
	//(*agent).sight[1][2] = map_g1[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).sight[2][0] = map_g1[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).sight[2][1] = map_g1[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).sight[2][2] = map_g1[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("to g1 next info > %d\n", (*agent).next_sight_g1[0][0]);
}

void next_agent_sight_g2(Agent *agent)
{

	(*agent).next_sight_g2[0][0] = 0;

	//(*agent).sight[0][0] = map_g1[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).sight[0][1] = map_g1[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).sight[0][2] = map_g1[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).sight[1][0] = map_g1[(*agent).location_x][(*agent).location_y - 1];
	(*agent).next_sight_g2[0][0] = map_g2[(*agent).location_x][(*agent).location_y];
	//(*agent).sight[1][2] = map_g1[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).sight[2][0] = map_g1[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).sight[2][1] = map_g1[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).sight[2][2] = map_g1[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("to g1 next info > %d\n", (*agent).next_sight_g2[0][0]);
}

void agent_action_select(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE]) //移動後のますが"W_info_math"かを判定し、もしそうだった場合再度乱数を回す。
{
	Wall_judge = 0;
	int z = (*agent).agent_action_select;
	switch (z)
	{

	case 0:
		if (map_g1[(*agent).location_x - 1][(*agent).location_y] != W_info_math)
		{
			break;
		}
		else
		{
			Wall_judge++;
		}
	case 1:
		if (map_g1[(*agent).location_x][(*agent).location_y + 1] != W_info_math)
		{
			break;
		}
		else
		{
			Wall_judge++;
		}
	case 2:
		if (map_g1[(*agent).location_x][(*agent).location_y - 1] != W_info_math)
		{
			break;
		}
		else
		{
			Wall_judge++;
		}
	case 3:
		if (map_g1[(*agent).location_x + 1][(*agent).location_y] != W_info_math)
		{
			break;
		}
		else
		{
			Wall_judge++;
		}
	}
}

void agent_action_dc(Agent *agent)
{
	int i = (*agent).agent_action_select;

	if (Wall_judge == 1)
	{
		printf("エージェントはその場に止まる\n");
		return;
	}
	else
	{
		switch (i)
		{
		case 0:
			printf("エージェントは上へ動く\n");
			(*agent).location_x = (*agent).location_x - 1;
			break;
		case 1:
			printf("エージェントは右へ動く\n");
			(*agent).location_y = (*agent).location_y + 1;
			break;
		case 2:
			printf("エージェントは左へ動く\n");
			(*agent).location_y = (*agent).location_y - 1;
			break;
		case 3:
			printf("エージェントは下へ動く\n");
			(*agent).location_x = (*agent).location_x + 1;
			break;
		}
	}

	(*agent).step_count++;
}

void init_rewards_post(Agent *agent)
{
	for (int i = 0; i < NUM_STEPS; i++)
	{
		(*agent).rewards[i] = 0;
	}
}

int goal_steps(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE], int map_g2[MAP_SIZE][MAP_SIZE])
{
	int a = 1;
	int b = 0;
	if ((*agent).step_count < NUM_CHANGE)
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
			printf("%d\n", a);
			return a;
		}
		else
		{
			printf("%d\n", b);
			return b;
		}
	}
}

void rewards_post(Agent *agent, int c, int isample2)
{
	(*agent).rewards[isample2] += c;
}

void ReadData(Agent *agent, int isample2)
{
	int sight_g1, sight_g2;
	sight_g1 = (*agent).sight_g1[0][0];
	sight_g2 = (*agent).sight_g2[0][0];

	if (sight_g1 == 1)
	{
		tx[isample2][0] = 1;
		tx[isample2][1] = 0;
		tx[isample2][2] = 0;
		tx[isample2][3] = 0;
		tx[isample2][4] = 0;
	}

	if (sight_g1 == 2)
	{
		tx[isample2][0] = 0;
		tx[isample2][1] = 1;
		tx[isample2][2] = 0;
		tx[isample2][3] = 0;
		tx[isample2][4] = 0;
	}

	if (sight_g1 == 3)
	{
		tx[isample2][0] = 0;
		tx[isample2][1] = 0;
		tx[isample2][2] = 1;
		tx[isample2][3] = 0;
		tx[isample2][4] = 0;
	}

	if (sight_g1 == 4)
	{
		tx[isample2][0] = 0;
		tx[isample2][1] = 0;
		tx[isample2][2] = 0;
		tx[isample2][3] = 1;
		tx[isample2][4] = 0;
	}

	if (sight_g1 == 5)
	{
		tx[isample2][0] = 0;
		tx[isample2][1] = 0;
		tx[isample2][2] = 0;
		tx[isample2][3] = 0;
		tx[isample2][4] = 1;
	}

	if (sight_g2 == 1)
	{
		tx[isample2][5] = 1;
		tx[isample2][6] = 0;
		tx[isample2][7] = 0;
		tx[isample2][8] = 0;
		tx[isample2][9] = 0;
	}

	if (sight_g2 == 2)
	{
		tx[isample2][5] = 0;
		tx[isample2][6] = 1;
		tx[isample2][7] = 0;
		tx[isample2][8] = 0;
		tx[isample2][9] = 0;
	}

	if (sight_g2 == 3)
	{
		tx[isample2][5] = 0;
		tx[isample2][6] = 0;
		tx[isample2][7] = 1;
		tx[isample2][8] = 0;
		tx[isample2][9] = 0;
	}

	if (sight_g2 == 4)
	{
		tx[isample2][5] = 0;
		tx[isample2][6] = 0;
		tx[isample2][7] = 0;
		tx[isample2][8] = 1;
		tx[isample2][9] = 0;
	}

	if (sight_g2 == 5)
	{
		tx[isample2][5] = 0;
		tx[isample2][6] = 0;
		tx[isample2][7] = 0;
		tx[isample2][8] = 0;
		tx[isample2][9] = 1;
	}
}

void NextReadData(Agent *agent, int isample2)
{
	int sight_g1, sight_g2;
	sight_g1 = (*agent).next_sight_g1[0][0];
	sight_g2 = (*agent).next_sight_g2[0][0];

	if (sight_g1 == 1)
	{
		next_tx[isample2][0] = 1;
		next_tx[isample2][1] = 0;
		next_tx[isample2][2] = 0;
		next_tx[isample2][3] = 0;
		next_tx[isample2][4] = 0;
	}

	if (sight_g1 == 2)
	{
		next_tx[isample2][0] = 0;
		next_tx[isample2][1] = 1;
		next_tx[isample2][2] = 0;
		next_tx[isample2][3] = 0;
		next_tx[isample2][4] = 0;
	}

	if (sight_g1 == 3)
	{
		next_tx[isample2][0] = 0;
		next_tx[isample2][1] = 0;
		next_tx[isample2][2] = 1;
		next_tx[isample2][3] = 0;
		next_tx[isample2][4] = 0;
	}

	if (sight_g1 == 4)
	{
		next_tx[isample2][0] = 0;
		next_tx[isample2][1] = 0;
		next_tx[isample2][2] = 0;
		next_tx[isample2][3] = 1;
		next_tx[isample2][4] = 0;
	}

	if (sight_g1 == 5)
	{
		next_tx[isample2][0] = 0;
		next_tx[isample2][1] = 0;
		next_tx[isample2][2] = 0;
		next_tx[isample2][3] = 0;
		next_tx[isample2][4] = 1;
	}

	if (sight_g2 == 1)
	{
		next_tx[isample2][5] = 1;
		next_tx[isample2][6] = 0;
		next_tx[isample2][7] = 0;
		next_tx[isample2][8] = 0;
		next_tx[isample2][9] = 0;
	}

	if (sight_g2 == 2)
	{
		next_tx[isample2][5] = 0;
		next_tx[isample2][6] = 1;
		next_tx[isample2][7] = 0;
		next_tx[isample2][8] = 0;
		next_tx[isample2][9] = 0;
	}

	if (sight_g2 == 3)
	{
		next_tx[isample2][5] = 0;
		next_tx[isample2][6] = 0;
		next_tx[isample2][7] = 1;
		next_tx[isample2][8] = 0;
		next_tx[isample2][9] = 0;
	}

	if (sight_g2 == 4)
	{
		next_tx[isample2][5] = 0;
		next_tx[isample2][6] = 0;
		next_tx[isample2][7] = 0;
		next_tx[isample2][8] = 1;
		next_tx[isample2][9] = 0;
	}

	if (sight_g2 == 5)
	{
		next_tx[isample2][5] = 0;
		next_tx[isample2][6] = 0;
		next_tx[isample2][7] = 0;
		next_tx[isample2][8] = 0;
		next_tx[isample2][9] = 1;
	}
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

void Feedforward(int isample2)
{
	int i, j;
	double net_input;

	// 順方向の動作
	// 訓練データに従って、ネットワークへの入力を設定する
	for (i = 0; i < NUM_INPUT + NUM_CON; i++)
	{
		if (i < NUM_INPUT)
		{
			x[i] = tx[isample2][i];
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

void NextFeedforward(int isample2)
{
	int i, j;
	double net_input;

	// 順方向の動作
	// 訓練データに従って、ネットワークへの入力を設定する
	for (i = 0; i < NUM_INPUT + NUM_CON; i++)
	{
		if (i < NUM_INPUT)
		{
			x[i] = next_tx[isample2][i];
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

void Backward(int isample2, Agent *agent)
{
	int i, j, n;
	double net_input;
	double alpha = ALPHA;
	double gamma = GAMMA;

	for (n = 0; n < NUM_OUTPUT; n++)
	{
		ty[isample2][n] = y[n] + (alpha * ((*agent).rewards[isample2] + gamma * next_maxq - y[n]));
	}

	// 逆方向の動作。
	// 出力層素子の逆伝搬時の動作。
	for (j = 0; j < NUM_OUTPUT; j++)
	{
		y_back[j] = BETA * (y[j] - ty[isample2][j]) * ((double)1.0 - y[j]) * y[j];
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
	double epsilon = (double)EPSILON_2;

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

void Infoselect(int isample2, Agent *agent)
{
	double epsilon = EPSILON_2;
	int info_choice = 0;
	double i = 0.0;

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
		}
		else
		{
			info_choice = (*agent).sight_g2[0][0];
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
	int i, j, max_q, n;
	int num_q[4];
	int same_q[4];
	int same_q_action[4];
	int how_many_same_q;

	double epsilon = EPSILON_2;
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
				n = j;
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

		(*agent).agent_action_select = same_q_action[genrand_int32() % n];
	}
}

void Act_Learn(Agent *agent, int isample2)
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

	qt[(*agent).agent_action_select][Choice_info - 1] = q + (alpha * (*agent).rewards[isample2] + (gamma * num_q[m]) - q);
}