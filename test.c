#include <stdio.h>
#include <stdlib.h>

int main(){
   char str[] = "111000000000000000000";

   long long result = strtoll(str,NULL,2);

   printf("%lld\n",result);
}