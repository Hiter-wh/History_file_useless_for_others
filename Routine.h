#pragma once

#pragma once
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include <vector>
#include <windows.h>

using namespace std;

#define ERROR_ 0
#define OK_ 1
#define MAX_routine_num 102//（24-7）*6，默认7小时睡眠,12:00~7:00
#define MAX_maxtrix_num 100
#define infinity_ 999
#define MAX_position_num 100
#define ACTIVITY_type int
#define ENTERTAINMENT 1
#define WORKING 0
typedef int Status;
//定义Routine
typedef struct Routine_temp {//定义活动

	float  begin, end;//开始，结束
	string position;
	int sequence;//序号
	int weight;//权重
	struct Routine_temp* prior;
	struct Routine_temp* next;
	string name;

}Routine_Node, * Routine;
//定义Day
typedef struct DAY {//定义链表
	Routine front;
	Routine rear;
	char* appendix;

}*Day, Day_node;

//定义哈希表
typedef struct HASH_TABLE
{
	string abbreviation;
	char* appendix;
}Hash_table[MAX_position_num];
Hash_table table;
int total_position_fot_Map = 0;
float Matrix_for_map_cost[MAX_maxtrix_num][MAX_maxtrix_num];
string LESSON[5];
string TIME_for_lesson[5];
string POS_for_lesson[5];

string Abbreviations_for_map[100];//读入地点简写

//初始化循环链表 
Status InitList(Routine& L, string accommondation = "QG");

//销毁一个带头结点的双向循环链表,ok 
Status DestroyList(Routine& L);
//打印的时候变为时间表示形式,以及转换回来
string change_int_form_into_time(float time);
float  change_time_into_int_form(string string_in);
// 打印循环链表信息
Status Print_daily(const Routine& L);

//在第i个位置插入信息
Status ListInsert_compulsory(Routine& L, float begin, float end, int positon,string name);

//删除第i个位置的信息
Status ListDelete(Routine& L, int sequence, int& e);
Status Check_confilct_compulsory(Routine& L);
Routine Find_Sequecne_for_begin(Routine& L, float begin, float end);
void Insert_accident(Routine& L,string name);
void delete_accident_(Routine& L);
/*
地图及寻址录入头文件
*/
//根据输入的附录字符串选择事情发生的地点。
int Find_a_proper_place(char* string);



typedef int Status;
//从txt中读取内容
int read_in_Map(string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Map.txt");




//读取相应地点
int check_Hash(string input_string);
//计算哈希函数
void calculate_Hash();
// 初始化邻接矩阵
void initialize_Matrix(string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Map.txt");

//打印图
float calculate_and_show_path(int v, int u, int*, int show_or_not = 1);
float show_path(string, string, int show_or_not = 1);

/*
课表录入头文件
*/



//读入课表
Status Syllabus_read_in(int workday, string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Syllabus.txt");

//解决中文输出乱码的问题
string change_into_letter(const char* str);
//输入课表处理
string get_pos_of_string(string WORKDAY);
string processing_string(string WORKDAY);

void show_today_schedule();


//


typedef struct voluntary {
	float begin, end, duration;
	string previous_position;
}voluntary_acticities[MAX_routine_num], voluntary_node;

typedef struct T {
	int weight;
	string name;
}things_want_to_do[MAX_routine_num];

things_want_to_do THINGS;

voluntary_acticities ACTIVITY;

float change_hour_and_minute_into_int_form(int hour, int minute);
void insert_voluntary_routine_for_week();

void seek_out_spare_time(const Routine L, string accommodation = "QG");
void Print_activity();