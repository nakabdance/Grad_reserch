#include <stdio.h>
#include <stdlib.h>
#include "Mt.h"

typedef struct
{
	//現在の座標
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
#define NUM_LEARN 10000 //学習の回数
#define NUM_INPUT 8
#define NUM_OUTPUT 4
#define EPSILON 0.05	 		// 学習時の重み修正の程度を決める。
#define THRESHOLD_ERROR 0.01	// 学習誤差がこの値以下になるとプログラムは停止する。
#define BETA 0.8				// 非線形性の強さ

int qt[4][2][2][2][2][2][2][2][2];	//qテーブル
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
void init_q_values(int qt[4][2][2][2][2][2][2][2][2]);

int main(int argc, char *argv[])
 {
	Agent a;
	a.location_x = 1;
	a.location_y = 11;
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
		printf("%d回目の行動\n", (ilearn + 1));
		showmap(Smap, &a);
		agent_sight(&a);
		agent_direction(&a);
		printf("\n");
		agent_action_select(&a, Smap);
		agent_action_dc(&a);
		make_Smap(Smap);
		showmap(Smap, &a);

		if(a.location_x == 1 && a.location_y == 1)
		{
			break;
		}

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
	int i, j;
	for( i = 0; i < 13; i++)//map配列の初期化
	{
		for( j = 0; j < 13; j++)
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


/*
	for (i = 1; i < 12; i++)
	{
		Smap[1][i] = N_Moved_math;
		Smap[11][i] = N_Moved_math;
		Smap[i][1] = N_Moved_math;
		Smap[i][11] = N_Moved_math;
	}

	//マップの障害物の配置
	Smap[2][6] = N_Moved_math;
	Smap[4][3] = N_Moved_math;
	Smap[4][9] = N_Moved_math;
	Smap[5][2] = N_Moved_math;
	Smap[5][3] = N_Moved_math;
	Smap[5][4] = N_Moved_math;
	Smap[5][10] = N_Moved_math;
	Smap[5][9] = N_Moved_math;
	Smap[5][8] = N_Moved_math;
	Smap[6][2] = N_Moved_math;
	Smap[6][3] = N_Moved_math;
	Smap[6][4] = N_Moved_math;
	Smap[6][6] = N_Moved_math;
	Smap[6][10] = N_Moved_math;
	Smap[6][9] = N_Moved_math;
	Smap[6][8] = N_Moved_math;
	Smap[7][2] = N_Moved_math;
	Smap[7][3] = N_Moved_math;
	Smap[7][4] = N_Moved_math;
	Smap[7][10] = N_Moved_math;
	Smap[7][9] = N_Moved_math;
	Smap[7][8] = N_Moved_math;
	Smap[8][6] = N_Moved_math;
	Smap[10][3] = N_Moved_math;
	Smap[10][9] = N_Moved_math;
	*/
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

void agent_action_select(Agent *a, int Smap[13][13]) //移動後のますがWall_mathかを判定し、もしそうだった場合再度乱数を回す。
{
	while (1)
	{
		(*a).agent_action_select = genrand_int32() % 4;
		int i = (*a).agent_action_select;
		switch(i)
		{
    		case 0:
			printf("エージェントは上へ動く\n");
    		if(Smap[(*a).location_x - 1][(*a).location_y] != Wall_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 1:
			printf("エージェントは右へ動く\n");
			if(Smap[(*a).location_x][(*a).location_y + 1] != Wall_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 2:
			printf("エージェントは左へ動く\n");
			if(Smap[(*a).location_x][(*a).location_y - 1] != Wall_math)
			{
				break;
			}
			else
			{
				continue;
			}
			case 3:
			printf("エージェントは下へ動く\n");
			if(Smap[(*a).location_x + 1][(*a).location_y] != Wall_math)
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


void init_q_values(int qt[4][2][2][2][2][2][2][2][2])
{
	int i,j,k,a,b,c,d,e,f;

	for(i = 0; i < 3; i++ )
	{
		for(j = 0; j < 1; j++ )
		{
			for(k = 0; k < 1; k++ )
			{
				for(a = 0; a < 1; a++ )
				{
					for(b = 0; b < 1; b++ )
					{
						for(c = 0; c < 1; c++ )
						{
							for(d = 0; d < 1; d++ )
							{
								for(e = 0; e < 1; e++ )
								{
									for(f = 0; f < 1; f++ )
									{
									qt[i][j][k][a][b][c][d][e][f] = 0;
									}
								qt[i][j][k][a][b][c][d][e][f] = 0;
								}
							qt[i][j][k][a][b][c][d][e][f] = 0;
							}
						qt[i][j][k][a][b][c][d][e][f] = 0;
						}
					qt[i][j][k][a][b][c][d][e][f] = 0;
					}
				qt[i][j][k][a][b][c][d][e][f] = 0;
				}
			qt[i][j][k][a][b][c][d][e][f] = 0;
			}
		qt[i][j][k][a][b][c][d][e][f] = 0;
		}
	qt[i][j][k][a][b][c][d][e][f] = 0;
	}
}