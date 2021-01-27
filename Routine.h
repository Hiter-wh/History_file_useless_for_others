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
#define MAX_routine_num 102//��24-7��*6��Ĭ��7Сʱ˯��,12:00~7:00
#define MAX_maxtrix_num 100
#define infinity_ 999
#define MAX_position_num 100
#define ACTIVITY_type int
#define ENTERTAINMENT 1
#define WORKING 0
typedef int Status;
//����Routine
typedef struct Routine_temp {//����

	float  begin, end;//��ʼ������
	string position;
	int sequence;//���
	int weight;//Ȩ��
	struct Routine_temp* prior;
	struct Routine_temp* next;
	string name;

}Routine_Node, * Routine;
//����Day
typedef struct DAY {//��������
	Routine front;
	Routine rear;
	char* appendix;

}*Day, Day_node;

//�����ϣ��
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

string Abbreviations_for_map[100];//����ص��д

//��ʼ��ѭ������ 
Status InitList(Routine& L, string accommondation = "QG");

//����һ����ͷ����˫��ѭ������,ok 
Status DestroyList(Routine& L);
//��ӡ��ʱ���Ϊʱ���ʾ��ʽ,�Լ�ת������
string change_int_form_into_time(float time);
float  change_time_into_int_form(string string_in);
// ��ӡѭ��������Ϣ
Status Print_daily(const Routine& L);

//�ڵ�i��λ�ò�����Ϣ
Status ListInsert_compulsory(Routine& L, float begin, float end, int positon,string name);

//ɾ����i��λ�õ���Ϣ
Status ListDelete(Routine& L, int sequence, int& e);
Status Check_confilct_compulsory(Routine& L);
Routine Find_Sequecne_for_begin(Routine& L, float begin, float end);
void Insert_accident(Routine& L,string name);
void delete_accident_(Routine& L);
/*
��ͼ��Ѱַ¼��ͷ�ļ�
*/
//��������ĸ�¼�ַ���ѡ�����鷢���ĵص㡣
int Find_a_proper_place(char* string);



typedef int Status;
//��txt�ж�ȡ����
int read_in_Map(string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Map.txt");




//��ȡ��Ӧ�ص�
int check_Hash(string input_string);
//�����ϣ����
void calculate_Hash();
// ��ʼ���ڽӾ���
void initialize_Matrix(string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Map.txt");

//��ӡͼ
float calculate_and_show_path(int v, int u, int*, int show_or_not = 1);
float show_path(string, string, int show_or_not = 1);

/*
�α�¼��ͷ�ļ�
*/



//����α�
Status Syllabus_read_in(int workday, string path = "C:\\Users\\86153\\Desktop\\back_end programme_development\\Syllabus.txt");

//�������������������
string change_into_letter(const char* str);
//����α���
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