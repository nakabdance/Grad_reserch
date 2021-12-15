#include <stdio.h>
#include <stdlib.h>
#include "Mt.h"

typedef struct
{
	int location_x;
	int location_y;
	int sight[3][3];		//周囲８近傍の視野
	int direction[3][3];	//方向情報
	int agent_action_select;
	long Q_number;
} Agent;

#define Wall_math -1 //見えない壁判定のあるマス
#define N_Moved_math 0 //移動不可能ます
#define Y_Moved_math 1 //移動可能ます
#define T_info_math 2 //上情報
#define L_info_math 3 //左情報
#define B_info_math 4 //下情報
#define R_info_math 5 //右情報
#define agent_location 6 //エージェントのいる場所
#define Map_size 10 //マップの大きさ
#define NUM_LEARN 20 //学習の回数

int map[13][13];
int Smap[13][13]; //11*11の視覚用マップ配列
int Vmap[13][13]= //11*11の方向情報用マップ配列,配列宣言時でしかリストの挿入ができないためここで作る
	{
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
		{-1,0,0,0,0,0,0,0,0,0,0,0,-1},
		{-1,0,4,4,4,4,0,4,4,4,4,0,-1},
		{-1,0,5,5,5,5,3,3,3,3,3,0,-1},
		{-1,0,2,0,5,4,4,4,3,0,2,0,-1},
		{-1,0,0,0,0,4,3,3,0,0,0,0,-1},
		{-1,0,0,0,0,4,0,4,0,0,0,0,-1},
		{-1,0,0,0,0,4,3,4,0,0,0,0,-1},
		{-1,0,4,4,4,4,4,4,4,4,4,0,-1},
		{-1,0,4,3,3,3,3,3,3,3,3,0,-1},
		{-1,0,0,0,2,2,2,2,2,0,2,0,-1},
		{-1,0,0,0,0,0,0,0,0,0,0,0,-1},
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
	}; 

void showmap(int map[13][13], Agent *a); //マップを可視化する関数
void make_Smap(int Smap[13][13]);
void agent_sight(Agent *a);
void agent_direction(Agent *a);
void agent_action_dc(Agent *a);		//エージェントの行動決定を司る関数
void agent_action_select(Agent *a, int Smap[13][13]);		//エージェントの行動選択を司る関数

int main(int argc, char *argv[])
 {
	Agent a;
	a.location_x = 1;
	a.location_y = 9;
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	int ilearn;

	init_genrand(Mtseed);

	make_Smap(Smap);
	showmap(Smap, &a);

	printf("\n\n");

	for(ilearn=0; ilearn<NUM_LEARN; ilearn++)
	{
		make_Smap(Smap);
		printf("%d回目の行動", (ilearn + 1));
		agent_sight(&a);
		agent_direction(&a);
		printf("\n");
		agent_action_select(&a, Smap);
		agent_action_dc(&a);
		showmap(Smap, &a);

	}
	return(0);
}

void showmap(int map[13][13], Agent *a)//map表示用関数、map[0][0]...には見えない壁があるので1から12までを表示する
{ 
	int mapi = 0;
	int mapn = 0;
	map[(*a).location_x][(*a).location_y] = agent_location;
	for( mapi = 1; mapi < 12; mapi++ )
	{
		for( mapn = 1; mapn < 12; mapn++ )
		{
			printf("%d.", map[mapi][mapn]);
		}
		printf("\n");
	}
}

void make_Smap(int Smap[13][13])
{
	int i, j; //map配列初期化
	for( i = 0; i < 11; i++)//map配列の初期化
	{
		for( j = 0; j < 11; j++)
		{
			Smap[i][j] = Y_Moved_math;
		}
	}

	for (i = 0; i < 13; i++)
	{
		Smap[0][i] = Wall_math;
		Smap[12][i] = Wall_math;
		Smap[i][0] = Wall_math;
		Smap[i][12] = Wall_math;
	}


	for (i = 1; i < 12; i++)
	{
		Smap[1][i] = N_Moved_math;
		Smap[11][i] = N_Moved_math;
		Smap[i][1] = N_Moved_math;
		Smap[i][11] = N_Moved_math;
	}

	//マップの障害物の配置
	Smap[1][5] = N_Moved_math;
	Smap[3][2] = N_Moved_math;
	Smap[3][8] = N_Moved_math;
	Smap[4][1] = N_Moved_math;
	Smap[4][2] = N_Moved_math;
	Smap[4][3] = N_Moved_math;
	Smap[4][9] = N_Moved_math;
	Smap[4][8] = N_Moved_math;
	Smap[4][7] = N_Moved_math;
	Smap[5][1] = N_Moved_math;
	Smap[5][2] = N_Moved_math;
	Smap[5][3] = N_Moved_math;
	Smap[5][5] = N_Moved_math;
	Smap[5][9] = N_Moved_math;
	Smap[5][8] = N_Moved_math;
	Smap[5][7] = N_Moved_math;
	Smap[6][1] = N_Moved_math;
	Smap[6][2] = N_Moved_math;
	Smap[6][3] = N_Moved_math;
	Smap[6][9] = N_Moved_math;
	Smap[6][8] = N_Moved_math;
	Smap[6][7] = N_Moved_math;
	Smap[7][5] = N_Moved_math;
	Smap[9][2] = N_Moved_math;
	Smap[9][8] = N_Moved_math;
}

void agent_sight(Agent *a)
{
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			(*a).sight[i][j] = 0;
		}
	}
	(*a).sight[0][0] = Smap[(*a).location_x - 1][(*a).location_y - 1];
	(*a).sight[0][1] = Smap[(*a).location_x - 1][(*a).location_y];
	(*a).sight[0][2] = Smap[(*a).location_x - 1][(*a).location_y + 1];
	(*a).sight[1][0] = Smap[(*a).location_x][(*a).location_y - 1];
	(*a).sight[1][1] = Smap[(*a).location_x][(*a).location_y];
	(*a).sight[1][2] = Smap[(*a).location_x][(*a).location_y + 1];
	(*a).sight[2][0] = Smap[(*a).location_x + 1][(*a).location_y - 1];
	(*a).sight[2][1] = Smap[(*a).location_x + 1][(*a).location_y];
	(*a).sight[2][2] = Smap[(*a).location_x + 1][(*a).location_y + 1];

	printf("\n");

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			printf("%d.", (*a).sight[i][j]);
		}
		printf("\n");
	}
}

void agent_direction(Agent *a)
{
	int i, j;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			(*a).direction[i][j] = 0;
		}
	}
	(*a).direction[0][0] = Vmap[(*a).location_x - 1][(*a).location_y - 1];
	(*a).direction[0][1] = Vmap[(*a).location_x - 1][(*a).location_y];
	(*a).direction[0][2] = Vmap[(*a).location_x - 1][(*a).location_y + 1];
	(*a).direction[1][0] = Vmap[(*a).location_x][(*a).location_y - 1];
	(*a).direction[1][1] = Vmap[(*a).location_x][(*a).location_y];
	(*a).direction[1][2] = Vmap[(*a).location_x][(*a).location_y + 1];
	(*a).direction[2][0] = Vmap[(*a).location_x + 1][(*a).location_y - 1];
	(*a).direction[2][1] = Vmap[(*a).location_x + 1][(*a).location_y];
	(*a).direction[2][2] = Vmap[(*a).location_x + 1][(*a).location_y + 1];

	printf("\n");

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			printf("%d.", (*a).direction[i][j]);
		}
		printf("\n");
	}
}

void agent_action_select(Agent *a, int Smap[13][13])
{
	while (1)
	{
		if (Smap[(*a).location_x][(*a).location_y] != Wall_math)
		{
			(*a).agent_action_select = genrand_int32() % 4;
			break;
		}
	}
	
}

void agent_action_dc(Agent *a)
{
	int i = (*a).agent_action_select;

	switch(i)
	{
    case 0:
    (*a).location_x = (*a).location_x - 1;
      break;
    case 1:
    (*a).location_y = (*a).location_y + 1;
      break;
    case 2:
    (*a).location_y = (*a).location_y - 1;
      break;
	case 3:
    (*a).location_x = (*a).location_x + 1;
      break;
  	}
}

