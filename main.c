#include <stdio.h>
#include <stdlib.h>
#include "Mt.h"

#define Y_Moved_math 1 //移動可能ます
#define N_Moved_math 0 //移動不可能ます
#define T_info_math 2 //上情報
#define L_info_math 3 //左情報
#define B_info_math 4 //下情報
#define R_info_math 5 //右情報
#define Map_size 10 //マップの大きさ

int Smap[11][11]; //11*11のマップ用配列

void showmap(int map[11][11]); //マップを可視化する関数
void make_Smap(int map[11][11]);

int main(int argc, char *argv[])
 {
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	init_genrand(Mtseed);

	make_Smap(Smap);
	showmap(Smap);
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

void make_Smap(int map[11][11])
{
	int i, j; //map配列初期化
	for( i = 0; i < 11; i++)//map配列の初期化
	{
		for( j = 0; j < 11; j++)
		{
			map[i][j] = Y_Moved_math;
		}
	}

	for (i = 0; i < Map_size+1; i++)
	{
		map[0][i] = N_Moved_math;
		map[Map_size][i] = N_Moved_math;
		map[i][0] = N_Moved_math;
		map[i][Map_size] = N_Moved_math;
	}

	//マップの障害物の配置
	map[1][5] = N_Moved_math;
	map[3][2] = N_Moved_math;
	map[3][8] = N_Moved_math;
	map[4][1] = N_Moved_math;
	map[4][2] = N_Moved_math;
	map[4][3] = N_Moved_math;
	map[4][9] = N_Moved_math;
	map[4][8] = N_Moved_math;
	map[4][7] = N_Moved_math;
	map[5][1] = N_Moved_math;
	map[5][2] = N_Moved_math;
	map[5][3] = N_Moved_math;
	map[5][5] = N_Moved_math;
	map[5][9] = N_Moved_math;
	map[5][8] = N_Moved_math;
	map[5][7] = N_Moved_math;
	map[6][1] = N_Moved_math;
	map[6][2] = N_Moved_math;
	map[6][3] = N_Moved_math;
	map[6][9] = N_Moved_math;
	map[6][8] = N_Moved_math;
	map[6][7] = N_Moved_math;
	map[7][5] = N_Moved_math;
	map[9][2] = N_Moved_math;
	map[9][8] = N_Moved_math;
}

void make_Vmap(int map[11][11])
{
	int map[11][11] = 
	{
		{0,0,0,0,0,0,0,0,0,0,0,},
		{0,4,4,4,4,0,4,4,4,4,0,},
		{0,5,5,5,5,3,3,3,3,3,0,},
		{0,2,0,5,4,4,4,3,0,2,0,},
		{0,0,0,0,4,3,3,0,0,0,0,},
		{0,0,0,0,4,0,4,0,0,0,0,},
		{0,0,0,0,4,3,4,0,0,0,0,},
		{0,4,4,4,4,4,4,4,4,4,0,},
		{0,4,3,3,3,3,3,3,3,3,0,},
		{0,0,0,2,2,2,2,2,0,2,0,},
		{0,0,0,0,0,0,0,0,0,0,0,},
	};
}



