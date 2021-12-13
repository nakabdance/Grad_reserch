#include <stdio.h>
#include <stdlib.h>
#include "Mt.h"

typedef struct
{
	int location_x;
	int location_y;
	int sight[3][3];		//周囲８近傍の視野
	int direction[3][3];	//方向情報
	unsigned long Q_number;
	int action_dc;
} Agent;

#define N_Moved_math 0 //移動不可能ます
#define Y_Moved_math 1 //移動可能ます
#define T_info_math 2 //上情報
#define L_info_math 3 //左情報
#define B_info_math 4 //下情報
#define R_info_math 5 //右情報
#define agent_location 6 //エージェントのいる場所
#define Map_size 10 //マップの大きさ

int map[11][11];
int Smap[11][11]; //11*11の視覚用マップ配列
int Vmap[11][11]= //11*11の方向情報用マップ配列,配列宣言時でしかリストの挿入ができないためここで作る
	{
		{0,0,0,0,0,0,0,0,0,0,0},
		{0,4,4,4,4,0,4,4,4,4,0},
		{0,5,5,5,5,3,3,3,3,3,0},
		{0,2,0,5,4,4,4,3,0,2,0},
		{0,0,0,0,4,3,3,0,0,0,0},
		{0,0,0,0,4,0,4,0,0,0,0},
		{0,0,0,0,4,3,4,0,0,0,0},
		{0,4,4,4,4,4,4,4,4,4,0},
		{0,4,3,3,3,3,3,3,3,3,0},
		{0,0,0,2,2,2,2,2,0,2,0},
		{0,0,0,0,0,0,0,0,0,0,0}
	}; 

void showmap(int map[11][11]); //マップを可視化する関数
void make_Smap(int Smap[11][11]);
void agent_sight(Agent *a);
void agent_direction(Agent *a);

int main(int argc, char *argv[])
 {
	Agent a;
	a.location_x = 1;
	a.location_y = 9;
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	init_genrand(Mtseed);

	make_Smap(Smap);
	showmap(Smap);

	printf("\n\n");

	showmap(Vmap);

agent_sight(&a);

agent_direction(&a);

	return(0);
}

void showmap(int map[11][11])//map表示用関数
{ 
	int mapi = 0;
	int mapn = 0;
	for( mapi = 0; mapi < 11; mapi++ )
	{
		for( mapn = 0; mapn < 11; mapn++ )
		{
			printf("%d.", map[mapi][mapn]);
		}
		printf("\n");
	}
}

void make_Smap(int Smap[11][11])
{
	int i, j; //map配列初期化
	for( i = 0; i < 11; i++)//map配列の初期化
	{
		for( j = 0; j < 11; j++)
		{
			Smap[i][j] = Y_Moved_math;
		}
	}

	for (i = 0; i < Map_size+1; i++)
	{
		Smap[0][i] = N_Moved_math;
		Smap[Map_size][i] = N_Moved_math;
		Smap[i][0] = N_Moved_math;
		Smap[i][Map_size] = N_Moved_math;
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
