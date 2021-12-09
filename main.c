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

int map[11][11]; //11*11のマップ用配列

void showmap(int map[11][11]); //マップを可視化する関数
void make_map(int map[11][11]);

int main(int argc, char *argv[])
 {
	unsigned long int Mtseed;
	Mtseed = strtoul(argv[1], NULL, 10);
	init_genrand(Mtseed);

	make_map(map);
	showmap(map);
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

void make_map(int map[11][11])
{
	int i, j; //map配列初期化
	for( i = 0; i < 11; i++)//map配列の初期化
	{
		for( j = 0; j < 11; j++)
		{
			map[i][j] = 1;
		}
	}

	for (i = 0; i < Map_size+1; i++)
	{
		map[0][i] = 0;
		map[Map_size][i] = 0;
		map[i][0] = 0;
		map[i][Map_size] = 0;
	}
}


