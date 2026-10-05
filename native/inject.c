/* Load native/uncrashed_spawn.dll into the running shipping game.
   Uses the Win32 loader inside the same Proton prefix. No stealth. */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

static DWORD find_game(void) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W entry;
    DWORD pid = 0;
    if (snap == INVALID_HANDLE_VALUE) {
        return 0;
    }
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snap, &entry)) {
        do {
            if (wcsstr(entry.szExeFile, L"Uncrashed-Win64-Shipping.exe")) {
                pid = entry.th32ProcessID;
                break;
            }
        } while (Process32NextW(snap, &entry));
    }
    CloseHandle(snap);
    return pid;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 2) {
        fwprintf(stderr, L"usage: inject.exe <path-to-uncrashed_spawn.dll>\n");
        return 2;
    }
    const wchar_t *dll = argv[1];
    DWORD pid = find_game();
    HANDLE process;
    void *remote;
    HANDLE thread;
    SIZE_T bytes = (wcslen(dll) + 1) * sizeof(wchar_t);
    DWORD code = 0;

    if (!pid) {
        fwprintf(stderr, L"shipping exe is not in this prefix\n");
        return 1;
    }
    wprintf(L"pid %lu\n", pid);
    process = OpenProcess(
        PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
        FALSE,
        pid);
    if (!process) {
        fwprintf(stderr, L"OpenProcess %lu\n", GetLastError());
        return 1;
    }
    remote = VirtualAllocEx(process, NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote || !WriteProcessMemory(process, remote, dll, bytes, NULL)) {
        fwprintf(stderr, L"write path %lu\n", GetLastError());
        return 1;
    }
    thread = CreateRemoteThread(process, NULL, 0, (LPTHREAD_START_ROUTINE)LoadLibraryW, remote, 0, NULL);
    if (!thread) {
        fwprintf(stderr, L"CreateRemoteThread %lu\n", GetLastError());
        return 1;
    }
    WaitForSingleObject(thread, 8000);
    GetExitCodeThread(thread, &code);
    wprintf(L"LoadLibrary %lu\n", code);
    CloseHandle(thread);
    CloseHandle(process);
    return code ? 0 : 2;
}
