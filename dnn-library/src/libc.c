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
  while (1)
    ;
}

void __assert_func(const char *file, int line, const char *func, const char *failedexpr)
{
  printf("assertion \"%s\" failed: file \"%s\", line %d%s%s\n",
    failedexpr, file, line, func ? ", function: " : "", func ? func : "");
  abort();
}
