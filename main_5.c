#include <stdio.h>
#include <stdlib.h>
#include "Mt.h"

typedef struct
{
	//現在の座標
	int location_x;
	int location_y;
	int sight[1][1];		//周囲の視野
	int direction[1][1];	//方向情報
	int agent_action_select; //移動方向を保持する
} Agent;

#define W_info_math 0 //壁情報
#define R_info_math 1 //右情報
#define T_info_math 2 //上情報
#define L_info_math 3 //左情報
#define D_info_math 4 //下情報
#define G_info_math 5 //ゴール情報
#define agent_location 6 //エージェントのいる場所

#define MAP_SIZE 11 //マップの大きさ
#define NUM_LEARN 1000 //学習の回数
#define NUM_STEPS 10 //エージェントの動ける回数
#define NUM_INPUT 8
#define NUM_OUTPUT 4
#define EPSILON 0.1	 // 探索率
#define ALPHA 0.1 //学習率
#define GAMMA  0.90 //割引率


int map[MAP_SIZE][MAP_SIZE];
int map_g1[MAP_SIZE][MAP_SIZE] = //ゴール１をめざす
{		
	{0,0,0,0,0,0,0,0,0,0,0},
	{0,5,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,2,3,3,3,3,3,3,3,3,0},
	{0,0,0,0,0,0,0,0,0,0,0}
}; 
int map_g2[MAP_SIZE][MAP_SIZE]= //ゴール２を目指すマップ
{	
	{0,0,0,0,0,0,0,0,0,0,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,4,3,3,3,3,3,3,3,3,0},
	{0,5,3,3,3,3,3,3,3,3,0},
	{0,0,0,0,0,0,0,0,0,0,0}
}; 

void showmap(int map[MAP_SIZE][MAP_SIZE], Agent *agent); //マップを可視化する関数
//void make_Smap(int map_g1[MAP_SIZE][MAP_SIZE]);
void agent_sight(Agent *agent);
void agent_direction(Agent *agent);
void agent_action_dc(Agent *agent);		//エージェントの行動決定を司る関数
void agent_action_select(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE]);		//エージェントの状態遷移
void episode_end(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE], int map_g2[MAP_SIZE][MAP_SIZE]); //ゴールに辿り着いてエピソードが終了したか判断する


int main(int argc, char *argv[])
 {
	Agent agent;
	agent.location_x = 4;
	agent.location_y = 8;
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	int ilearn;
	int steps;

	init_genrand(Mtseed);

	//make_Smap(map_g1);
	showmap(map_g1, &agent);

	printf("\n\n");

	for(ilearn=0; ilearn<NUM_LEARN; ilearn++)
	{
		agent.location_x = 4;
		agent.location_y = 8;
		for (steps=0; steps<NUM_STEPS; steps++)
		{
			printf("%d回目の行動", (steps + 1));
			agent_sight(&agent);
			agent_direction(&agent);
			printf("\n");
			agent_action_select(&agent, map_g1);
			agent_action_dc(&agent);

			showmap(map_g1, &agent);

		}

	}
	return(0);
}

void showmap(int map_g1[MAP_SIZE][MAP_SIZE], Agent *agent)//map表示用関数
{ 
	int mapi = 0;
	int mapn = 0;
	int Smap[MAP_SIZE][MAP_SIZE];

	for( mapi = 0; mapi < MAP_SIZE; mapi++ )
	{
		for( mapn = 0; mapn < MAP_SIZE; mapn++ )
		{
			Smap[mapi][mapn] = map_g1[mapi][mapn];
		}
	}

	Smap[(*agent).location_x][(*agent).location_y] = agent_location;

	for( mapi = 0; mapi < MAP_SIZE; mapi++ )
	{
		for( mapn = 0; mapn < MAP_SIZE; mapn++ )
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

void agent_sight(Agent *agent)
{

	(*agent).sight[0][0] = 0;

	//(*agent).sight[0][0] = map_g1[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).sight[0][1] = map_g1[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).sight[0][2] = map_g1[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).sight[1][0] = map_g1[(*agent).location_x][(*agent).location_y - 1];
	(*agent).sight[0][0] = map_g1[(*agent).location_x][(*agent).location_y];
	//(*agent).sight[1][2] = map_g1[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).sight[2][0] = map_g1[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).sight[2][1] = map_g1[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).sight[2][2] = map_g1[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("\n");


	printf("to g1 info > %d", (*agent).sight[0][0]);

}

void agent_direction(Agent *agent)
{

	(*agent).direction[0][0] = 0;

	//(*agent).direction[0][0] = map_g2[(*agent).location_x - 1][(*agent).location_y - 1];
	//(*agent).direction[0][1] = map_g2[(*agent).location_x - 1][(*agent).location_y];
	//(*agent).direction[0][2] = map_g2[(*agent).location_x - 1][(*agent).location_y + 1];
	//(*agent).direction[1][0] = map_g2[(*agent).location_x][(*agent).location_y - 1];
	(*agent).direction[0][0] = map_g2[(*agent).location_x][(*agent).location_y];
	//(*agent).direction[1][2] = map_g2[(*agent).location_x][(*agent).location_y + 1];
	//(*agent).direction[2][0] = map_g2[(*agent).location_x + 1][(*agent).location_y - 1];
	//(*agent).direction[2][1] = map_g2[(*agent).location_x + 1][(*agent).location_y];
	//(*agent).direction[2][2] = map_g2[(*agent).location_x + 1][(*agent).location_y + 1];

	printf("\n");

	printf("to g2 info > %d", (*agent).direction[0][0]);
}

void agent_action_select(Agent *agent, int map_g1[MAP_SIZE][MAP_SIZE]) //移動後のますが"W_info_math"かを判定し、もしそうだった場合再度乱数を回す。
{
	while (1)
	{
		(*agent).agent_action_select = genrand_int32() % 4;
		int z = (*agent).agent_action_select;
		switch(z)
		{
    		case 0:
    		if(map_g1[(*agent).location_x - 1][(*agent).location_y] != W_info_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 1:
			if(map_g1[(*agent).location_x][(*agent).location_y + 1] != W_info_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 2:
			if(map_g1[(*agent).location_x][(*agent).location_y - 1] != W_info_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 3:
			if(map_g1[(*agent).location_x + 1][(*agent).location_y] != W_info_math)
			{
				break;
			}
			else
			{
				continue;
			}
  		}
		break;
	}
}

void agent_action_dc(Agent *agent)
{
	int i = (*agent).agent_action_select;

	switch(i)
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