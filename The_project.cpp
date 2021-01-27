
#include "Routine.h"
using namespace std;
int main(void)
{
	//初始化及地图、课表录入
	Day today = new Day_node;
	int routines;
	int workday;
	int flag = 1;
	//总功能测试
	total_position_fot_Map = read_in_Map();
	calculate_Hash();
	initialize_Matrix();
	while (flag)
	{
		cout << "今天是周：" << endl;
		cin >> workday;
		if (Syllabus_read_in(workday) == OK_)flag = 0;
	}
	InitList(today->front);
	Check_confilct_compulsory(today->front);
	Print_daily(today->front);
	//插入
	cout << "请输入想插入的活动的名称" << endl;
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
//将TIME_for_lesson里的字符串变为可用的时间
float  change_time_into_int_form(string string_in) {
	float total = 0;
	if ((string_in[0] == '0') && ((string_in[1] - '0') < 7)) {
		cout << "错误的作息时间！" << endl;return 0;
	}
	else
	{
		for (int k = 0;k < 5;k++) {
			if (((string_in[k]) <= '9') && ((string_in[k]) >= '0') || (string_in[k] == ':'));
			//cout<<"success!";
			else
			{
				cout << "错误输入" << endl;
				return 0;
			}
		}

		total = (string_in[0] - '0') * 10 * 6 + (string_in[1] - '0') * 6;
		total += ((long int)string_in[3] - '0') + 0.1 * ((long int)string_in[4] - '0');
		total -= 42;

	}
	return total;
}
//将时间变为字符串输出
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

//初始化循环链表,正序创建节点，输入必要活动,需要先导入课表,done
Status InitList(Routine& L, string accommodation) {//输入头节点
	Routine temporary, s;//中间变量  
	//头节点初始化
	int n = 0;
	int sequence = 0;
	L = new Routine_Node();
	L->next = L->prior = L;
	L->sequence = sequence;
	L->position = accommodation;
	if (!TIME_for_lesson[1][1]) { cout << "未录入课表，无必要活动" << endl;return ERROR_; }
	//计算课表里的必要活动数目
	for (int i = 1;i <= 4;i++) { if (LESSON[i][0] != '\\')n++; }
	cout << "共" << n << "个必要活动" << endl;

	temporary = L;
	for (int i = 1;i <= 4;i++)
	{
		if (LESSON[i][0] != '\\') {
			//cout << "放入" << i << endl;
			s = new Routine_Node;
			if (!s) exit(0);
			sequence++;
			cout << TIME_for_lesson[i] << endl;
			s->begin = change_time_into_int_form(TIME_for_lesson[i]);
			//cout <<"第"<<sequence<<"件事开始于"<< s->begin << endl;
			s->end = s->begin + 10.5;
			//if (s->begin > s->end) { cout << "错误输入，重新输入！" << endl; i--;delete(s);continue; }

			//cout << "重新给出此活动在你心中的分量(1~10)" << endl;//输入权重
			s->weight = 10;
			//cout << "给出此活动发生的地点" << endl;
			s->position = POS_for_lesson[i];
			s->name = LESSON[i];
			if (s->end > (MAX_routine_num)) { cout << "活动过多！停止输入" << endl;break; }//活动过多则不在录入
			//若活动数目在正常范围内，则给每个活动添加序号

			s->sequence = sequence;
			//指针s向下扩增
			temporary->next = s;
			s->prior = temporary;
			temporary = s;
		}
	}

	temporary->next = L;//连接头尾节点
	L->prior = temporary;
	cout << "输入了" << sequence << "个活动" << endl;
	Print_daily(L);
	return OK_;


}


//销毁双向链表,done
Status  DestroyList(Routine& L)//销毁一个带头结点的双向循环链表,OK_  
{
	//temporary接收下一个数据地址	 
	Routine temporary = NULL;

	while (L)
	{
		temporary = L->next;
		delete L;
		L = temporary;
	}
	return OK_;
}


// 打印循环链表信息,done
Status Print_daily(const Routine& L) {
	if (L->next == L && L->prior == L) {//只有头节点
		cout << "双向链表为空,错误！" << endl;
	}

	Routine temporary = L->next;

	while (temporary != L)
	{
		cout << "从" << change_int_form_into_time( temporary->begin) << "开始，到" << change_int_form_into_time(temporary->end )<< endl;
		cout << "这是今天的第" << temporary->sequence << "个活动" << endl;
		cout << "发生地点为" << temporary->position << endl;
		cout << "名字为" << temporary->name << endl;
		
		if ((temporary->position == temporary->prior->position) || (temporary==L->next));
		else
		{
			cout << "路径为：" << endl;
			show_path(temporary->prior->position, temporary->position, 1);
		}
			temporary = temporary->next;
	}

	cout << "输出完毕" << endl;
	return OK_;
}

//判断是否非空,done
Status is_Empty(Routine L)
{
	if (L->next == L)return OK_;
	else
		return ERROR_;
}

//检测基本活动是否有冲突并处理,done
Status Check_confilct_compulsory(Routine& L) //L为头指针
{
	if (is_Empty(L) == OK_) { cout << "无活动" << endl; }
	Routine temp = L->next;
	Routine temp_2;
	int weight_1 = 0;
	int weight_2 = 0;
	int remain = 0;
	int i = 0;
	//cout << "开始检测冲突" << endl;
	while (temp->next != L)
	{
		int cost_on_the_way = 0;
		int flag = 1;
		//先检测是否超时
		if (temp->end > MAX_routine_num)
		{
			cout << "时间规划总和超过17小时，之后活动一律删除" << endl;
			temp->prior->next = L;
		}
		if (temp->position == temp->next->position) { cost_on_the_way = 0; }
		else
		{
			//cout << "开始计算路径花费" << endl;
			cost_on_the_way = show_path(temp->position, temp->next->position, 0);
			//cout << "路径花费计算完毕" << endl;
		}

		//无冲突时
		if ((temp->end + cost_on_the_way) < temp->next->begin)
		{
			cout << "第" << temp->sequence << "件事与第" << temp->next->sequence << "件事无冲突" << endl;

		}
		else
		{//冲突处理
			if ((temp->end + cost_on_the_way - temp->next->begin) > ((temp->next->end - temp->next->begin) / 2))
			{
				cout << "序号为" << temp->sequence << "的活动与活动" << temp->next->sequence << "产生严重冲突,请输入两件事的权重，第" << temp->sequence << "件事的权重:" << endl;
				cin >> weight_1;
				cout << "第" << temp->next->sequence << "件事的权重:" << endl;
				cin >> weight_2;
				if (weight_1 < weight_2)
				{
					//当删除temp之后会丢失其后继节点，所以用temp_2保存
					temp_2 = temp->next;
					ListDelete(L, temp->sequence, remain);
					cout << "原开始时间为" << remain << "的活动因冲突已经删除" << endl;
					temp = temp_2;flag = 0;
				}
				else
				{
					ListDelete(L, temp->next->sequence, remain);
					cout << "原开始时间为" << remain << "的活动因冲突已经删除" << endl;
				}

			}
			else
			{
				cout << "活动" << temp->sequence << "与活动" << temp->next->sequence << "冲突存在但在可接受范围内，继续执行" << endl;
			}
		}
		i++;
	//	cout << "检测了" << i << "次" << endl;
		if (flag)
			temp = temp->next;
	}
	//cout << "冲突检测完毕" << endl;
	return OK_;

}


//查询双向链表中的第i个数据，并返回给e,undo
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

//根据活动开始时间查找到它之前的一个活动，done
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
				//准确定位插入活动的地点
				if (temporary->next->begin >= begin) { break; }
				else { temporary = temporary->next; }
				break;
			}
			cout << "找到插入位置" << endl;
		}
	}
	return temporary;

}


//在序号为i的位置后插入数据信息,并更新序号,done
Status ListInsert_compulsory(Routine& L, float begin, float end, int positon,string name) {
	Routine  temporary, will_be_inserted;
	if (begin >= end) { cout << "时间无效输入！" << endl;return ERROR_; }
	//先获取其之前的活动的位置，传给temporary 
	temporary = Find_Sequecne_for_begin(L, begin, end);
	cout << "已获取插入位置，从原序号为" << temporary->sequence << "之后的位置开始插入" << endl;
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
	//开始更新序号
	cout << "开始更新序号" << endl;
	will_be_inserted->sequence = temporary->sequence + 1;
	temporary = temporary->next->next;
	while (temporary != L)
	{
		temporary->sequence++;
		temporary = temporary->next;
	}
	cout << "插入完成！" << endl;
	return OK_;

}


//删除序号为i的活动，并更新序号,done
Status ListDelete(Routine& L, int sequence, int& e) {
	Routine temporary, temp;
	//判断链表非空
	if (is_Empty(L) == OK_) { cout << "已经没有活动需要执行了" << endl; }
	temporary = L->next;
	//查找目标节点
	while ((temporary->sequence != sequence) && (temporary != L))
	{
		temporary = temporary->next;
	}
	if (temporary == L) { cout << "无此序号所代表的活动" << endl;return ERROR_; }
	//找到后删除，并修改后续节点序号
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
	cout << "现在输入想插入活动的地点,英文输入分号:" << endl;
	string in_put;
	cin >> in_put;
	string begin, end;
	cout << "现在输入插入活动的开始时间：" << endl;
	cin >> begin;
	cout << "现在输入插入活动的结束时间：" << endl;
	cin >> end;
	ListInsert_compulsory(L, change_time_into_int_form( begin),change_time_into_int_form( end), check_Hash(in_put),name);

}
void delete_accident_(Routine& L)
{
	int temp;
	int remain;
	cout << "输入你想删除的活动序号" << endl;
	cin >> temp;
	ListDelete(L, temp, remain);
}
/*
****
以上为Routine
基本操作部分//2190120405黎之旭
以下为地图及课表自动录入部分
****
*/

//读入地图
int read_in_Map(string path)
{


	int i;
	int calculate = 0;
	int repeated_y, repeated_x;

	//获取字符串
	fstream  file_in(path);
	char  line[1024] = { 0 };
	string  x = "未";
	string  y = "录";//初始化
	string  z = "入";
	cout << "开始录入地图，以下为工大地图" << endl;
	while (file_in.getline(line, sizeof(line)))
	{
		repeated_x = 0;
		repeated_y = 0;
		i = 0;
		//计算不相同的字符串总数，用来设置哈希表
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
	cout << "共" << calculate << "个不同的地点" << endl;
	if (calculate == 0) { cout << "未输入正确的地图地址" << endl; }
	file_in.clear();
	file_in.close();
	return calculate;
}

//计算哈希
void calculate_Hash()
{
	string temp;
	int hash_num;
	int flag = 1;
	int problem = 0;
	bool used[100] = { false };
	//计算哈希
	for (int i = 0;i < total_position_fot_Map;i++) {
		int flag = 1;

		temp = Abbreviations_for_map[i];
		hash_num = (temp[0] - 'A') + (temp[1] - 'A');
		hash_num = hash_num % total_position_fot_Map;
		//cout << endl << "哈希地址为" << hash_num << endl;
		while (flag) {
			//cout << used[hash_num] << endl;
			if (used[hash_num] == false)
			{
				table[hash_num].abbreviation = temp;
				flag = 0;
				//	cout << "将" << temp << "存储于" << hash_num << endl;
				used[hash_num] = true;

			}
			//冲突处理 hash_num 在0~34之间
			else
			{
				hash_num++;hash_num = hash_num % total_position_fot_Map;
				//problem++;
				//if (problem > total_position_fot_Map)return ;
			}
		}
	}
	cout << "哈希读取准备完毕" << endl;
}

//检查哈希函数
int check_Hash(string input_string)
{
	int hash_num, flag = 1;
	int count = 0;
	hash_num = (input_string[0] - 'A') + (input_string[1] - 'A');
	hash_num = hash_num % total_position_fot_Map;
	//cout << "理论上地址为" << hash_num << endl;
	while (flag)
	{
		if (table[hash_num].abbreviation == input_string) { return hash_num;flag = 0; }
		else
		{
			//cout << "没找到" << endl;
			hash_num++;hash_num = hash_num % total_position_fot_Map;
			count++;
			if (count > 40) { cout << "错误的地址输入！" << endl;return ERROR_; }
		}
	}
}

//初始化矩阵
void initialize_Matrix(string path)
{
	//因为为无向图，所以对称初始化
	for (int i = 0;i < 100;i++)
		for (int k = 0;k < 100;k++)
		{
			Matrix_for_map_cost[i][k] = infinity_;
		}

	//cout << "存储矩阵初始化完毕" << endl;
	//获取字符串
	fstream  file_in(path);
	char  line[1024] = { 0 };
	string  x = "未";
	string  y = "录";//初始化
	float z = infinity_;
	//cout << "开始录入数据" << endl;
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
	//cout << "矩阵数据录入完毕" << endl;

}

//用Djastra求路径
void Djastra(int v, float* distence, int* previous_path)//2维数组不写成指针
{

	//cout << "开始计算路径" << endl;
	//for (int a= 0;  a<MAX_maxtrix_num; a++)


	bool visited[100];//bool类型数组，默认全为0；bool visited[10]={1},默认全为1.visited[i]表示i是否放入visited中，1表示放入。
	//对distence,previous_path的初始化。
	for (int i = 1; i <= total_position_fot_Map; i++)
	{
		distence[i] = Matrix_for_map_cost[v][i];//初始距离
		visited[i] = 0;//一开始没有任何节点放入
		if (distence[i] == infinity_)//distence=infinity_表示两节点不相连
			previous_path[i] = 0;//故前项节点为0；
		else previous_path[i] = v;//如果相连，前项节点为v

	}
	distence[v] = 0; visited[v] = 1;//将起点距离设为0，并将其放入visited;

	//遍历所有点，将他们加入visited
	for (int i = 2; i <= total_position_fot_Map; i++)
	{
		int tmp = infinity_;
		int u = v;
		//找到离起点1最近的第二点
		for (int j = 1; j <= total_position_fot_Map; j++)//找最近点u
		{
			if ((!visited[j]) && distence[j] < tmp)
			{
				u = j;
				tmp = distence[j];
			}
		}
		visited[u] = 1;//将标号为u的点放入；
		//更新距离distence
		for (int i = 1; i <= total_position_fot_Map; i++)
		{
			if ((!visited[i]) && Matrix_for_map_cost[u][i] < infinity_)//以放入的u为中间点。
			{
				int newdist = distence[u] + Matrix_for_map_cost[u][i];
				if (newdist < distence[i])
				{
					distence[i] = newdist;//更新距离为如1->2->4
					previous_path[i] = u;//4的前项节点为i;
				}

			}
		}

	}
	//cout << "路径算法计算完毕" << endl;
}
//打印路径
float calculate_and_show_path(int v, int u, int* previous_path, int show_or_not)
{

	int queue[100] = { 0 };
	int total = 1;
	queue[total] = u;//u是最后放入的节点
	total++;
	int tmp = previous_path[u];//u的前项节点
	while (tmp != v)//直到找到起始点v
	{
		queue[total] = tmp;
		total++;
		//cout << "total is"  <<total << endl;
		//cout << "上一个地点的哈希地址为" << tmp << endl;
		tmp = previous_path[tmp];

	}
	queue[total] = v;
	if (show_or_not) {
		for (int i = total; i >= 1; i--)//输出
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
	//if (show_or_not) { cout << "开始展示并计算路径" << endl; }
	Djastra(check_Hash(departure), distence, previous_path);
	//cout << "Djastra已计算完成" << endl;
	return calculate_and_show_path(check_Hash(departure), check_Hash(destination), previous_path, show_or_not);
}


/*
****
以上为地图的录入（地图在桌面上的TXT文件中）
及哈希，最短路径选择等部分//2190120405黎之旭
以下为课表自动录入及排序部分
****
*/
string change_into_letter(const char* str)
{
	string result;
	WCHAR* character_form;
	char* letter_form;

	//获得临时变量的大小
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	character_form = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, character_form, i);

	//获得临时变量的大小
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
	if ((workday < 1) || (workday > 5)) { cout << "错误的工作日输入！" << endl;return ERROR_; }
	string MON, TUE, WEN, TUR, FRI, TIME;
	string MON_POS, TUE_POS, WEN_POS, TUR_POS, FRI_POS;
	fstream file_in(path);
	std::string line;
	cout << "工大学子课程表：" << endl;
	string title;
	getline(file_in, title);
	if (!title[0]) { cout << "未找到正确存储路径地址" << endl;return ERROR_; }
	title = change_into_letter(title.c_str());
	cout << title << endl;
	//开始录入周一至周五课表并打印
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

		//字符串整合
		MON_POS = get_pos_of_string(MON);
		MON = processing_string(MON);
		cout << MON << " ，地点为" << MON_POS << ends;


		TUE_POS = get_pos_of_string(TUE);
		TUE = processing_string(TUE);
		cout << TUE << " ，地点为" << TUE_POS << ends;


		WEN_POS = get_pos_of_string(WEN);
		WEN = processing_string(WEN);
		cout << WEN << " ，地点为" << WEN_POS << ends;

		TUR_POS = get_pos_of_string(TUR);
		TUR = processing_string(TUR);
		cout << TUR << " ，地点为" << TUR_POS << ends;

		FRI_POS = get_pos_of_string(FRI);
		FRI = processing_string(FRI);
		cout << FRI << " ，地点为" << FRI_POS << endl;
		//记录时间
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
			cout << "标记点" << endl;
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
		if (LESSON[i][0] != '\\') { cout << "在" << TIME_for_lesson[i] << "于" << POS_for_lesson[i] << "上" << LESSON[i] << "课" << endl; }

		i++;
	}
}




/*
****
以上为基本功能实现
下面的为自主时间规划//2190120405黎之旭
PS：由于时间问题，我们只能缩短在这个部分的工期，但是我们还是会尽力把这个功能做好，
我们的目标是:让这个桌面应用能够真正的帮助到同学们，而非应付差事。
****
*/
//输出显示转换
float change_hour_and_minute_into_int_form(int hour, int minute)
{
	float time;
	if ((hour >= 7) && (hour <= 24));else { cout << "时间错误输入！" << endl; return 0; }
	if ((hour == 24) && (minute == 0));else { cout << "时间错误输入！" << endl; return 0; }
	if ((minute < 60) && (minute > 0));else { cout << "时间错误输入！" << endl; return 0; }
	time = (hour - 7) * 6 + (minute / 10);
	return time;
}

//寻找剩余时间
void seek_out_spare_time(const Routine L, string accommodation)
{
	int i = 0;Routine temp;

	ACTIVITY[0].begin = 0;
	ACTIVITY[0].previous_position = accommodation;
	if (L->next == L) { cout << "全部时间为空余时间" << endl;ACTIVITY[0].end = MAX_routine_num;return; }
	else ACTIVITY[0].end = L->next->begin;
	//cout << "开始首相录入" << endl;
	//cout << accommodation << L->next->position;
	ACTIVITY[0].duration = (ACTIVITY[0].end - ACTIVITY[0].begin) - show_path(accommodation, L->next->position, 0);
	temp = L->next;
	//首项检验
	if (ACTIVITY[0].begin >= ACTIVITY[0].end) { cout << "醒来无空余时间" << endl;i = 0; }
	else i++;
	//开始录入空余时间
	if (temp->next != L)
	{
		for (i;i <= (L->prior->sequence - 1);i++) {
			ACTIVITY[i].begin = temp->end;
			ACTIVITY[i].previous_position = temp->position;
			ACTIVITY[i].end = temp->next->begin;
			int cost_on_the_way_;
			//cout << "开始计算应当减去的时间"<<i << endl;
			if (temp->position == temp->next->position) { cost_on_the_way_ = 0; }
			else { cost_on_the_way_ = show_path(temp->position, temp->next->position, 0); }
			ACTIVITY[i].duration = (ACTIVITY[i].end - ACTIVITY[i].begin) - cost_on_the_way_;
			temp = temp->next;
		}
		//末项处理
		if ((temp->next == L) && (temp->end != MAX_routine_num)) {
			ACTIVITY[i].begin = temp->end;
			ACTIVITY[i].previous_position = temp->position;
			ACTIVITY[i].end = MAX_routine_num;
		}
		else
		{
			cout << "出现错误！" << endl;
		}
	}

	cout << "空余时间寻找完毕" << endl;
}
//打印输入的活动
void Print_activity()
{
	int i = 0;
	for (i;(ACTIVITY[i].end) < MAX_routine_num;i++) { cout << "空余时间为：从" << change_int_form_into_time(ACTIVITY[i].begin) << "  到" << change_int_form_into_time(ACTIVITY[i].end) << endl; }
	cout << "空余时间为：从" << change_int_form_into_time(ACTIVITY[i].begin) << "  到" << change_int_form_into_time(ACTIVITY[i].end) << endl;
}
//输入空余时间想做的事
void insert_voluntary_routine_for_week()
{
 things_want_to_do things;
	string position;
	int weight = 1;
	cout << "请输入您还想做的事及它们的权重" << endl;
	for (int i = 0;weight != 0;i++)
	{
		cout << "请输入第" << i + 1 << "件事的权重与名称,先输入权重(权重为0则结束)：" << endl;
		cin >> weight;
		cout << "现在输入名称:" << endl;
		cin >> position;
		int flag = 1;
		//cout << position << endl;
		int k = i;
		cout << k << endl;
		 		while (flag)
			{

				if (k == 0) { things[k].name = position;things[k].weight = weight; flag = 0;cout << "已找到插入位置并插入" << endl; }
				else
				{
					if (weight > things[k - 1].weight)
					{

						things[k].weight = things[k - 1].weight;things[k].name = things[k - 1].name;
						cout << things[k].name << things[k].weight << endl;
						k--;
						//	cout << "前移" << endl;
					}
					else
					{
						things[k].weight = weight;things[k].name = position;flag = 0;cout << "已找到插入位置并插入" << endl;
					}
				}
			}
		}

	for (int j = 0;things[j].weight!= 0;j++)
		cout << "建议" << j + 1 << "件事,做" << things[j].name << endl;
}






