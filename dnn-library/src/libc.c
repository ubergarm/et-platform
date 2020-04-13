#include <stdint.h>
#include <stdio.h>
#include <syscall.h>

void *memset(void *s, int c, size_t n)
{
  char *p = s;

  while (n) {
    *p++ = c;
    n--;
  }

  return s;
}

void *memcpy(void *dest, const void *src, size_t n)
{
  const char *s = src;
  char *d = dest;

  while (n) {
    *d++ = *s++;
    n--;
  }

  return dest;
}

int memcmp(const void *s1, const void *s2, size_t n)
{
  unsigned char u1, u2;

  for (; n--; s1++, s2++) {
    u1 = *(unsigned char *)s1;
    u2 = *(unsigned char *)s2;
    if (u1 != u2)
      return u1 - u2;
  }

  return 0;
}

size_t strlen(const char *str)
{
  const char *s = str;

  while (*s)
    s++;

  return s - str;
}

int putchar(int c)
{
  syscall(SYSCALL_LOG_WRITE, (uint64_t)&c, 1, 0);
  return c;
}

int puts(const char *s)
{
  size_t len = strlen(s);
  syscall(SYSCALL_LOG_WRITE, (uint64_t)s, len, 0);
  putchar('\n');
  return (int)len;
}

void abort(void)
{
  puts("Aborted.\n");
  // and assembly code to tell sysemu/vcs/zebu to stop the test with a failure
  __asm__ __volatile__
    (
     "fence\n"
     "slti x0,x0,0x7ff\n"
     "lui t0, 0x50BAD\n"
     "csrw validation0, t0\n"
     "wfi\n"
     : : : "t0");
  
  while (1)
    ;
}

void __assert_func(const char *file, int line, const char *func, const char *failedexpr)
{
  printf("Assertion \"%s\" failed: file \"%s\", line %d%s%s\n",
         failedexpr, file, line, func ? ", function: " : "", func ? func : "");
  abort();
}
