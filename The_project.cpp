
#include "Routine.h"
using namespace std;
int main(void)
{
	//��ʼ������ͼ���α�¼��
	Day today = new Day_node;
	int routines;
	int workday;
	int flag = 1;
	//�ܹ��ܲ���
	total_position_fot_Map = read_in_Map();
	calculate_Hash();
	initialize_Matrix();
	while (flag)
	{
		cout << "�������ܣ�" << endl;
		cin >> workday;
		if (Syllabus_read_in(workday) == OK_)flag = 0;
	}
	InitList(today->front);
	Check_confilct_compulsory(today->front);
	Print_daily(today->front);
	//����
	cout << "�����������Ļ������" << endl;
	string accidently_insert;
	cin >> accidently_insert;
	Insert_accident(today->front, accidently_insert);
	Check_confilct_compulsory(today->front);
	Print_daily(today->front);
	delete_accident_(today->front);
	Print_daily(today->front);
	seek_out_spare_time(today->front);
	Print_activity();
	insert_voluntary_routine_for_week();

}
//��TIME_for_lesson����ַ�����Ϊ���õ�ʱ��
float  change_time_into_int_form(string string_in) {
	float total = 0;
	if ((string_in[0] == '0') && ((string_in[1] - '0') < 7)) {
		cout << "�������Ϣʱ�䣡" << endl;return 0;
	}
	else
	{
		for (int k = 0;k < 5;k++) {
			if (((string_in[k]) <= '9') && ((string_in[k]) >= '0') || (string_in[k] == ':'));
			//cout<<"success!";
			else
			{
				cout << "��������" << endl;
				return 0;
			}
		}

		total = (string_in[0] - '0') * 10 * 6 + (string_in[1] - '0') * 6;
		total += ((long int)string_in[3] - '0') + 0.1 * ((long int)string_in[4] - '0');
		total -= 42;

	}
	return total;
}
//��ʱ���Ϊ�ַ������
string change_int_form_into_time(float time)
{
	string time_out, time_out1, time_out2;
	int  temp_1 = (int)(time / 6);
	int  temp_2 = (time - (temp_1) * 6) * 10;
	temp_1 += 7;
	stringstream temp1, temp2;
	temp1 << temp_1;
	temp2 << temp_2;
	temp1 >> time_out1;
	temp2 >> time_out2;
	if (temp_2 < 10) { time_out2 = "0" + time_out2; }
	time_out = time_out1 + ":" + time_out2;
	return time_out;
}

//��ʼ��ѭ������,���򴴽��ڵ㣬�����Ҫ�,��Ҫ�ȵ���α�,done
Status InitList(Routine& L, string accommodation) {//����ͷ�ڵ�
	Routine temporary, s;//�м����  
	//ͷ�ڵ��ʼ��
	int n = 0;
	int sequence = 0;
	L = new Routine_Node();
	L->next = L->prior = L;
	L->sequence = sequence;
	L->position = accommodation;
	if (!TIME_for_lesson[1][1]) { cout << "δ¼��α��ޱ�Ҫ�" << endl;return ERROR_; }
	//����α���ı�Ҫ���Ŀ
	for (int i = 1;i <= 4;i++) { if (LESSON[i][0] != '\\')n++; }
	cout << "��" << n << "����Ҫ�" << endl;

	temporary = L;
	for (int i = 1;i <= 4;i++)
	{
		if (LESSON[i][0] != '\\') {
			//cout << "����" << i << endl;
			s = new Routine_Node;
			if (!s) exit(0);
			sequence++;
			cout << TIME_for_lesson[i] << endl;
			s->begin = change_time_into_int_form(TIME_for_lesson[i]);
			//cout <<"��"<<sequence<<"���¿�ʼ��"<< s->begin << endl;
			s->end = s->begin + 10.5;
			//if (s->begin > s->end) { cout << "�������룬�������룡" << endl; i--;delete(s);continue; }

			//cout << "���¸����˻�������еķ���(1~10)" << endl;//����Ȩ��
			s->weight = 10;
			//cout << "�����˻�����ĵص�" << endl;
			s->position = POS_for_lesson[i];
			s->name = LESSON[i];
			if (s->end > (MAX_routine_num)) { cout << "����ֹ࣡ͣ����" << endl;break; }//���������¼��
			//�����Ŀ��������Χ�ڣ����ÿ���������

			s->sequence = sequence;
			//ָ��s��������
			temporary->next = s;
			s->prior = temporary;
			temporary = s;
		}
	}

	temporary->next = L;//����ͷβ�ڵ�
	L->prior = temporary;
	cout << "������" << sequence << "���" << endl;
	Print_daily(L);
	return OK_;


}


//����˫������,done
Status  DestroyList(Routine& L)//����һ����ͷ����˫��ѭ������,OK_  
{
	//temporary������һ�����ݵ�ַ	 
	Routine temporary = NULL;

	while (L)
	{
		temporary = L->next;
		delete L;
		L = temporary;
	}
	return OK_;
}


// ��ӡѭ��������Ϣ,done
Status Print_daily(const Routine& L) {
	if (L->next == L && L->prior == L) {//ֻ��ͷ�ڵ�
		cout << "˫������Ϊ��,����" << endl;
	}

	Routine temporary = L->next;

	while (temporary != L)
	{
		cout << "��" << change_int_form_into_time( temporary->begin) << "��ʼ����" << change_int_form_into_time(temporary->end )<< endl;
		cout << "���ǽ���ĵ�" << temporary->sequence << "���" << endl;
		cout << "�����ص�Ϊ" << temporary->position << endl;
		cout << "����Ϊ" << temporary->name << endl;
		
		if ((temporary->position == temporary->prior->position) || (temporary==L->next));
		else
		{
			cout << "·��Ϊ��" << endl;
			show_path(temporary->prior->position, temporary->position, 1);
		}
			temporary = temporary->next;
	}

	cout << "������" << endl;
	return OK_;
}

//�ж��Ƿ�ǿ�,done
Status is_Empty(Routine L)
{
	if (L->next == L)return OK_;
	else
		return ERROR_;
}

//��������Ƿ��г�ͻ������,done
Status Check_confilct_compulsory(Routine& L) //LΪͷָ��
{
	if (is_Empty(L) == OK_) { cout << "�޻" << endl; }
	Routine temp = L->next;
	Routine temp_2;
	int weight_1 = 0;
	int weight_2 = 0;
	int remain = 0;
	int i = 0;
	//cout << "��ʼ����ͻ" << endl;
	while (temp->next != L)
	{
		int cost_on_the_way = 0;
		int flag = 1;
		//�ȼ���Ƿ�ʱ
		if (temp->end > MAX_routine_num)
		{
			cout << "ʱ��滮�ܺͳ���17Сʱ��֮��һ��ɾ��" << endl;
			temp->prior->next = L;
		}
		if (temp->position == temp->next->position) { cost_on_the_way = 0; }
		else
		{
			//cout << "��ʼ����·������" << endl;
			cost_on_the_way = show_path(temp->position, temp->next->position, 0);
			//cout << "·�����Ѽ������" << endl;
		}

		//�޳�ͻʱ
		if ((temp->end + cost_on_the_way) < temp->next->begin)
		{
			cout << "��" << temp->sequence << "�������" << temp->next->sequence << "�����޳�ͻ" << endl;

		}
		else
		{//��ͻ����
			if ((temp->end + cost_on_the_way - temp->next->begin) > ((temp->next->end - temp->next->begin) / 2))
			{
				cout << "���Ϊ" << temp->sequence << "�Ļ��" << temp->next->sequence << "�������س�ͻ,�����������µ�Ȩ�أ���" << temp->sequence << "���µ�Ȩ��:" << endl;
				cin >> weight_1;
				cout << "��" << temp->next->sequence << "���µ�Ȩ��:" << endl;
				cin >> weight_2;
				if (weight_1 < weight_2)
				{
					//��ɾ��temp֮��ᶪʧ���̽ڵ㣬������temp_2����
					temp_2 = temp->next;
					ListDelete(L, temp->sequence, remain);
					cout << "ԭ��ʼʱ��Ϊ" << remain << "�Ļ���ͻ�Ѿ�ɾ��" << endl;
					temp = temp_2;flag = 0;
				}
				else
				{
					ListDelete(L, temp->next->sequence, remain);
					cout << "ԭ��ʼʱ��Ϊ" << remain << "�Ļ���ͻ�Ѿ�ɾ��" << endl;
				}

			}
			else
			{
				cout << "�" << temp->sequence << "��" << temp->next->sequence << "��ͻ���ڵ��ڿɽ��ܷ�Χ�ڣ�����ִ��" << endl;
			}
		}
		i++;
	//	cout << "�����" << i << "��" << endl;
		if (flag)
			temp = temp->next;
	}
	//cout << "��ͻ������" << endl;
	return OK_;

}


//��ѯ˫�������еĵ�i�����ݣ������ظ�e,undo
Status Select(const Routine& L, int i, int& e, int& b) {
	Routine temporary = L->next;

	while (temporary != L && i > 1) {
		i--;
		temporary = temporary->next;

	}

	b = temporary->begin;
	e = temporary->end;
	return OK_;
}

//���ݻ��ʼʱ����ҵ���֮ǰ��һ�����done
Routine Find_Sequecne_for_begin(Routine& L, float begin, float end)
{
	Routine temporary;
	temporary = L->next;
	if (begin > L->prior->end) { temporary = L->prior; }
	else {
		if (end < temporary->begin) { temporary = L; }
		else
		{
			while (temporary->next != L)
			{
				while (temporary->next->end < begin)
				{
					temporary = temporary->next;
				}
				//׼ȷ��λ�����ĵص�
				if (temporary->next->begin >= begin) { break; }
				else { temporary = temporary->next; }
				break;
			}
			cout << "�ҵ�����λ��" << endl;
		}
	}
	return temporary;

}


//�����Ϊi��λ�ú����������Ϣ,���������,done
Status ListInsert_compulsory(Routine& L, float begin, float end, int positon,string name) {
	Routine  temporary, will_be_inserted;
	if (begin >= end) { cout << "ʱ����Ч���룡" << endl;return ERROR_; }
	//�Ȼ�ȡ��֮ǰ�Ļ��λ�ã�����temporary 
	temporary = Find_Sequecne_for_begin(L, begin, end);
	cout << "�ѻ�ȡ����λ�ã���ԭ���Ϊ" << temporary->sequence << "֮���λ�ÿ�ʼ����" << endl;
	//if(temporary->e){}
	if (!(will_be_inserted = new Routine_Node())) return ERROR_;
	will_be_inserted->begin = begin;
	will_be_inserted->end = end;
	will_be_inserted->prior = temporary;
	will_be_inserted->next = temporary->next;
	temporary->next->prior = will_be_inserted;
	temporary->next = will_be_inserted;
	will_be_inserted->name = name;
	will_be_inserted->position = table[positon].abbreviation;
	//��ʼ�������
	cout << "��ʼ�������" << endl;
	will_be_inserted->sequence = temporary->sequence + 1;
	temporary = temporary->next->next;
	while (temporary != L)
	{
		temporary->sequence++;
		temporary = temporary->next;
	}
	cout << "������ɣ�" << endl;
	return OK_;

}


//ɾ�����Ϊi�Ļ�����������,done
Status ListDelete(Routine& L, int sequence, int& e) {
	Routine temporary, temp;
	//�ж�����ǿ�
	if (is_Empty(L) == OK_) { cout << "�Ѿ�û�л��Ҫִ����" << endl; }
	temporary = L->next;
	//����Ŀ��ڵ�
	while ((temporary->sequence != sequence) && (temporary != L))
	{
		temporary = temporary->next;
	}
	if (temporary == L) { cout << "�޴����������Ļ" << endl;return ERROR_; }
	//�ҵ���ɾ�������޸ĺ����ڵ����
	temporary->prior->next = temporary->next;
	temporary->next->prior = temporary->prior;
	temp = temporary->next;
	e = temporary->begin;
	delete temporary;
	while (temp != L)
	{
		temp->sequence--;
		temp = temp->next;

	}
	return OK_;
}

int Find_a_proper_place(char* string)
{
	int pos = 0;

	return pos;
}
void Insert_accident(Routine& L ,string name) {
	cout << "��������������ĵص�,Ӣ������ֺ�:" << endl;
	string in_put;
	cin >> in_put;
	string begin, end;
	cout << "������������Ŀ�ʼʱ�䣺" << endl;
	cin >> begin;
	cout << "������������Ľ���ʱ�䣺" << endl;
	cin >> end;
	ListInsert_compulsory(L, change_time_into_int_form( begin),change_time_into_int_form( end), check_Hash(in_put),name);

}
void delete_accident_(Routine& L)
{
	int temp;
	int remain;
	cout << "��������ɾ���Ļ���" << endl;
	cin >> temp;
	ListDelete(L, temp, remain);
}
/*
****
����ΪRoutine
������������//2190120405��֮��
����Ϊ��ͼ���α��Զ�¼�벿��
****
*/

//�����ͼ
int read_in_Map(string path)
{


	int i;
	int calculate = 0;
	int repeated_y, repeated_x;

	//��ȡ�ַ���
	fstream  file_in(path);
	char  line[1024] = { 0 };
	string  x = "δ";
	string  y = "¼";//��ʼ��
	string  z = "��";
	cout << "��ʼ¼���ͼ������Ϊ�����ͼ" << endl;
	while (file_in.getline(line, sizeof(line)))
	{
		repeated_x = 0;
		repeated_y = 0;
		i = 0;
		//���㲻��ͬ���ַ����������������ù�ϣ��
		stringstream word(line);
		word >> x;
		word >> y;
		word >> z;
		cout << x << "  " << y << "  " << z << endl;
		while (i < calculate)
		{
			if (Abbreviations_for_map[i] == x) { repeated_x = 1;break; }
			i++;
		}
		i = 0;
		while (i < calculate)
		{
			if (Abbreviations_for_map[i] == y) { repeated_y = 1; break; }
			i++;
		}
		if (!repeated_x) {
			Abbreviations_for_map[calculate] = x; calculate++;
		}
		if (!repeated_y) {
			Abbreviations_for_map[calculate] = y; calculate++;
		}
	}
	cout << "��" << calculate << "����ͬ�ĵص�" << endl;
	if (calculate == 0) { cout << "δ������ȷ�ĵ�ͼ��ַ" << endl; }
	file_in.clear();
	file_in.close();
	return calculate;
}

//�����ϣ
void calculate_Hash()
{
	string temp;
	int hash_num;
	int flag = 1;
	int problem = 0;
	bool used[100] = { false };
	//�����ϣ
	for (int i = 0;i < total_position_fot_Map;i++) {
		int flag = 1;

		temp = Abbreviations_for_map[i];
		hash_num = (temp[0] - 'A') + (temp[1] - 'A');
		hash_num = hash_num % total_position_fot_Map;
		//cout << endl << "��ϣ��ַΪ" << hash_num << endl;
		while (flag) {
			//cout << used[hash_num] << endl;
			if (used[hash_num] == false)
			{
				table[hash_num].abbreviation = temp;
				flag = 0;
				//	cout << "��" << temp << "�洢��" << hash_num << endl;
				used[hash_num] = true;

			}
			//��ͻ���� hash_num ��0~34֮��
			else
			{
				hash_num++;hash_num = hash_num % total_position_fot_Map;
				//problem++;
				//if (problem > total_position_fot_Map)return ;
			}
		}
	}
	cout << "��ϣ��ȡ׼�����" << endl;
}

//����ϣ����
int check_Hash(string input_string)
{
	int hash_num, flag = 1;
	int count = 0;
	hash_num = (input_string[0] - 'A') + (input_string[1] - 'A');
	hash_num = hash_num % total_position_fot_Map;
	//cout << "�����ϵ�ַΪ" << hash_num << endl;
	while (flag)
	{
		if (table[hash_num].abbreviation == input_string) { return hash_num;flag = 0; }
		else
		{
			//cout << "û�ҵ�" << endl;
			hash_num++;hash_num = hash_num % total_position_fot_Map;
			count++;
			if (count > 40) { cout << "����ĵ�ַ���룡" << endl;return ERROR_; }
		}
	}
}

//��ʼ������
void initialize_Matrix(string path)
{
	//��ΪΪ����ͼ�����ԶԳƳ�ʼ��
	for (int i = 0;i < 100;i++)
		for (int k = 0;k < 100;k++)
		{
			Matrix_for_map_cost[i][k] = infinity_;
		}

	//cout << "�洢�����ʼ�����" << endl;
	//��ȡ�ַ���
	fstream  file_in(path);
	char  line[1024] = { 0 };
	string  x = "δ";
	string  y = "¼";//��ʼ��
	float z = infinity_;
	//cout << "��ʼ¼������" << endl;
	while (file_in.getline(line, sizeof(line)))
	{
		stringstream word(line);
		word >> x;
		word >> y;
		word >> z;
		//	cout << z << endl;
		Matrix_for_map_cost[check_Hash(x)][check_Hash(y)] = z;
		Matrix_for_map_cost[check_Hash(y)][check_Hash(x)] = z;

	}
	//cout << "��������¼�����" << endl;

}

//��Djastra��·��
void Djastra(int v, float* distence, int* previous_path)//2ά���鲻д��ָ��
{

	//cout << "��ʼ����·��" << endl;
	//for (int a= 0;  a<MAX_maxtrix_num; a++)


	bool visited[100];//bool�������飬Ĭ��ȫΪ0��bool visited[10]={1},Ĭ��ȫΪ1.visited[i]��ʾi�Ƿ����visited�У�1��ʾ���롣
	//��distence,previous_path�ĳ�ʼ����
	for (int i = 1; i <= total_position_fot_Map; i++)
	{
		distence[i] = Matrix_for_map_cost[v][i];//��ʼ����
		visited[i] = 0;//һ��ʼû���κνڵ����
		if (distence[i] == infinity_)//distence=infinity_��ʾ���ڵ㲻����
			previous_path[i] = 0;//��ǰ��ڵ�Ϊ0��
		else previous_path[i] = v;//���������ǰ��ڵ�Ϊv

	}
	distence[v] = 0; visited[v] = 1;//����������Ϊ0�����������visited;

	//�������е㣬�����Ǽ���visited
	for (int i = 2; i <= total_position_fot_Map; i++)
	{
		int tmp = infinity_;
		int u = v;
		//�ҵ������1����ĵڶ���
		for (int j = 1; j <= total_position_fot_Map; j++)//�������u
		{
			if ((!visited[j]) && distence[j] < tmp)
			{
				u = j;
				tmp = distence[j];
			}
		}
		visited[u] = 1;//�����Ϊu�ĵ���룻
		//���¾���distence
		for (int i = 1; i <= total_position_fot_Map; i++)
		{
			if ((!visited[i]) && Matrix_for_map_cost[u][i] < infinity_)//�Է����uΪ�м�㡣
			{
				int newdist = distence[u] + Matrix_for_map_cost[u][i];
				if (newdist < distence[i])
				{
					distence[i] = newdist;//���¾���Ϊ��1->2->4
					previous_path[i] = u;//4��ǰ��ڵ�Ϊi;
				}

			}
		}

	}
	//cout << "·���㷨�������" << endl;
}
//��ӡ·��
float calculate_and_show_path(int v, int u, int* previous_path, int show_or_not)
{

	int queue[100] = { 0 };
	int total = 1;
	queue[total] = u;//u��������Ľڵ�
	total++;
	int tmp = previous_path[u];//u��ǰ��ڵ�
	while (tmp != v)//ֱ���ҵ���ʼ��v
	{
		queue[total] = tmp;
		total++;
		//cout << "total is"  <<total << endl;
		//cout << "��һ���ص�Ĺ�ϣ��ַΪ" << tmp << endl;
		tmp = previous_path[tmp];

	}
	queue[total] = v;
	if (show_or_not) {
		for (int i = total; i >= 1; i--)//���
		{
			if (i != 1)
			{
				cout << table[queue[i]].abbreviation << "->";
			}
			else {
				cout << table[queue[i]].abbreviation << endl;
			}

		}
	}
	float cost = 0;
	for (int i = total;i >= 2;i--) {
		cost = cost + Matrix_for_map_cost[queue[i]][queue[i - 1]];
		//cout << Matrix_for_map_cost[queue[i]][queue[i - 1]] << endl;
	}
	return cost;
}
float show_path(string departure, string destination, int show_or_not)
{
	float distence[MAX_maxtrix_num];
	int previous_path[MAX_maxtrix_num];
	//if (show_or_not) { cout << "��ʼչʾ������·��" << endl; }
	Djastra(check_Hash(departure), distence, previous_path);
	//cout << "Djastra�Ѽ������" << endl;
	return calculate_and_show_path(check_Hash(departure), check_Hash(destination), previous_path, show_or_not);
}


/*
****
����Ϊ��ͼ��¼�루��ͼ�������ϵ�TXT�ļ��У�
����ϣ�����·��ѡ��Ȳ���//2190120405��֮��
����Ϊ�α��Զ�¼�뼰���򲿷�
****
*/
string change_into_letter(const char* str)
{
	string result;
	WCHAR* character_form;
	char* letter_form;

	//�����ʱ�����Ĵ�С
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	character_form = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, character_form, i);

	//�����ʱ�����Ĵ�С
	i = WideCharToMultiByte(CP_ACP, 0, character_form, -1, NULL, 0, NULL, NULL);
	letter_form = new char[i + 1];

	WideCharToMultiByte(CP_ACP, 0, character_form, -1, letter_form, i, NULL, NULL);

	result = letter_form;
	delete[]character_form;
	delete[]letter_form;

	return result;
}
string get_pos_of_string(string WORKDAY)
{
	if (WORKDAY[0] != '\\') {
		WORKDAY = WORKDAY.substr(WORKDAY.length() - 2);
	}
	else WORKDAY = "\\";
	return WORKDAY;
}
string processing_string(string WORKDAY)
{
	if (WORKDAY[0] != '\\') {
		WORKDAY = WORKDAY.substr(0, (WORKDAY.length() - 2));
	}
	return WORKDAY;
}
Status Syllabus_read_in(int workday, string path)
{
	string time[10];
	int i = 1;
	if ((workday < 1) || (workday > 5)) { cout << "����Ĺ��������룡" << endl;return ERROR_; }
	string MON, TUE, WEN, TUR, FRI, TIME;
	string MON_POS, TUE_POS, WEN_POS, TUR_POS, FRI_POS;
	fstream file_in(path);
	std::string line;
	cout << "����ѧ�ӿγ̱�" << endl;
	string title;
	getline(file_in, title);
	if (!title[0]) { cout << "δ�ҵ���ȷ�洢·����ַ" << endl;return ERROR_; }
	title = change_into_letter(title.c_str());
	cout << title << endl;
	//��ʼ¼����һ������α���ӡ
	while (getline(file_in, line)) {

		string str = change_into_letter(line.c_str()).c_str();
		//cout << str << endl;
		stringstream word(str);
		word >> time[i];
		word >> MON;
		word >> TUE;
		word >> WEN;
		word >> TUR;
		word >> FRI;
		cout << time[i] << ends;

		//�ַ�������
		MON_POS = get_pos_of_string(MON);
		MON = processing_string(MON);
		cout << MON << " ���ص�Ϊ" << MON_POS << ends;


		TUE_POS = get_pos_of_string(TUE);
		TUE = processing_string(TUE);
		cout << TUE << " ���ص�Ϊ" << TUE_POS << ends;


		WEN_POS = get_pos_of_string(WEN);
		WEN = processing_string(WEN);
		cout << WEN << " ���ص�Ϊ" << WEN_POS << ends;

		TUR_POS = get_pos_of_string(TUR);
		TUR = processing_string(TUR);
		cout << TUR << " ���ص�Ϊ" << TUR_POS << ends;

		FRI_POS = get_pos_of_string(FRI);
		FRI = processing_string(FRI);
		cout << FRI << " ���ص�Ϊ" << FRI_POS << endl;
		//��¼ʱ��
		TIME_for_lesson[i] = time[i];


		if (workday == 1) {
			LESSON[i] = MON;
			POS_for_lesson[i] = MON_POS;
		}
		else if (workday == 2) {
			LESSON[i] = TUE;
			POS_for_lesson[i] = TUE_POS;
		}
		else if (workday == 3) {
			LESSON[i] = WEN;POS_for_lesson[i] = WEN_POS;
		}
		else if (workday == 4) {
			LESSON[i] = TUR; POS_for_lesson[i] = TUR_POS;
		}
		else if (workday == 5) {
			LESSON[i] = FRI; POS_for_lesson[i] = FRI_POS;
			cout << "��ǵ�" << endl;
		}
		else
		{
			return ERROR_;
		}
		i++;
	}return OK_;
}
void show_today_schedule() {
	int i = 1;
	while (i <= 4) {
		if (LESSON[i][0] != '\\') { cout << "��" << TIME_for_lesson[i] << "��" << POS_for_lesson[i] << "��" << LESSON[i] << "��" << endl; }

		i++;
	}
}




/*
****
����Ϊ��������ʵ��
�����Ϊ����ʱ��滮//2190120405��֮��
PS������ʱ�����⣬����ֻ��������������ֵĹ��ڣ��������ǻ��ǻᾡ��������������ã�
���ǵ�Ŀ����:���������Ӧ���ܹ������İ�����ͬѧ�ǣ�����Ӧ�����¡�
****
*/
//�����ʾת��
float change_hour_and_minute_into_int_form(int hour, int minute)
{
	float time;
	if ((hour >= 7) && (hour <= 24));else { cout << "ʱ��������룡" << endl; return 0; }
	if ((hour == 24) && (minute == 0));else { cout << "ʱ��������룡" << endl; return 0; }
	if ((minute < 60) && (minute > 0));else { cout << "ʱ��������룡" << endl; return 0; }
	time = (hour - 7) * 6 + (minute / 10);
	return time;
}

//Ѱ��ʣ��ʱ��
void seek_out_spare_time(const Routine L, string accommodation)
{
	int i = 0;Routine temp;

	ACTIVITY[0].begin = 0;
	ACTIVITY[0].previous_position = accommodation;
	if (L->next == L) { cout << "ȫ��ʱ��Ϊ����ʱ��" << endl;ACTIVITY[0].end = MAX_routine_num;return; }
	else ACTIVITY[0].end = L->next->begin;
	//cout << "��ʼ����¼��" << endl;
	//cout << accommodation << L->next->position;
	ACTIVITY[0].duration = (ACTIVITY[0].end - ACTIVITY[0].begin) - show_path(accommodation, L->next->position, 0);
	temp = L->next;
	//�������
	if (ACTIVITY[0].begin >= ACTIVITY[0].end) { cout << "�����޿���ʱ��" << endl;i = 0; }
	else i++;
	//��ʼ¼�����ʱ��
	if (temp->next != L)
	{
		for (i;i <= (L->prior->sequence - 1);i++) {
			ACTIVITY[i].begin = temp->end;
			ACTIVITY[i].previous_position = temp->position;
			ACTIVITY[i].end = temp->next->begin;
			int cost_on_the_way_;
			//cout << "��ʼ����Ӧ����ȥ��ʱ��"<<i << endl;
			if (temp->position == temp->next->position) { cost_on_the_way_ = 0; }
			else { cost_on_the_way_ = show_path(temp->position, temp->next->position, 0); }
			ACTIVITY[i].duration = (ACTIVITY[i].end - ACTIVITY[i].begin) - cost_on_the_way_;
			temp = temp->next;
		}
		//ĩ���
		if ((temp->next == L) && (temp->end != MAX_routine_num)) {
			ACTIVITY[i].begin = temp->end;
			ACTIVITY[i].previous_position = temp->position;
			ACTIVITY[i].end = MAX_routine_num;
		}
		else
		{
			cout << "���ִ���" << endl;
		}
	}

	cout << "����ʱ��Ѱ�����" << endl;
}
//��ӡ����Ļ
void Print_activity()
{
	int i = 0;
	for (i;(ACTIVITY[i].end) < MAX_routine_num;i++) { cout << "����ʱ��Ϊ����" << change_int_form_into_time(ACTIVITY[i].begin) << "  ��" << change_int_form_into_time(ACTIVITY[i].end) << endl; }
	cout << "����ʱ��Ϊ����" << change_int_form_into_time(ACTIVITY[i].begin) << "  ��" << change_int_form_into_time(ACTIVITY[i].end) << endl;
}
//�������ʱ����������
void insert_voluntary_routine_for_week()
{
 things_want_to_do things;
	string position;
	int weight = 1;
	cout << "�����������������¼����ǵ�Ȩ��" << endl;
	for (int i = 0;weight != 0;i++)
	{
		cout << "�������" << i + 1 << "���µ�Ȩ��������,������Ȩ��(Ȩ��Ϊ0�����)��" << endl;
		cin >> weight;
		cout << "������������:" << endl;
		cin >> position;
		int flag = 1;
		//cout << position << endl;
		int k = i;
		cout << k << endl;
		 		while (flag)
			{

				if (k == 0) { things[k].name = position;things[k].weight = weight; flag = 0;cout << "���ҵ�����λ�ò�����" << endl; }
				else
				{
					if (weight > things[k - 1].weight)
					{

						things[k].weight = things[k - 1].weight;things[k].name = things[k - 1].name;
						cout << things[k].name << things[k].weight << endl;
						k--;
						//	cout << "ǰ��" << endl;
					}
					else
					{
						things[k].weight = weight;things[k].name = position;flag = 0;cout << "���ҵ�����λ�ò�����" << endl;
					}
				}
			}
		}

	for (int j = 0;things[j].weight!= 0;j++)
		cout << "����" << j + 1 << "����,��" << things[j].name << endl;
}






