{{-- Update x-data untuk menampung state modal baru --}}
<div x-data="{
    showPhotoModal: false,
    showUsernameModal: false,
    showPasswordModal: false,
    showDeleteModal: false
}"
x-on:username-updated.window="showUsernameModal = false"
x-on:photo-updated.window="showPhotoModal = false"
x-on:password-updated.window="showPasswordModal = false">

    <div class="max-w-4xl w-full mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden relative mb-10">

        {{-- Header Section (Tidak Berubah) --}}
        <div class="h-48 bg-gradient-to-r from-primary-100 to-primary-60 relative">
            <a href="javascript:history.back()" class="absolute top-8 left-8 flex items-center gap-2 px-5 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-full text-white hover:bg-white/20 transition group z-20">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 group-hover:-translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span class="text-smallBtn font-medium">Back</span>
            </a>
            <div class="absolute bottom-0 right-0 w-80 h-80 bg-accent-100 opacity-10 rounded-full translate-x-1/3 translate-y-1/3 blur-3xl pointer-events-none"></div>
            <div class="absolute top-0 right-20 w-40 h-40 bg-secondary-happy-100 opacity-10 rounded-full blur-2xl pointer-events-none"></div>
        </div>

        <div class="px-8 pb-12">
            {{-- Profile Photo & Save Button (Tidak Berubah) --}}
            <div class="flex flex-col md:flex-row justify-between items-end -mt-20 mb-10 gap-6 relative z-10">
                <div class="relative">
                    <div class="relative">
                        @if ($photo)
                            <img class="w-40 h-40 rounded-full border-[6px] border-white object-cover shadow-lg bg-shadedOfGray-20"
                                src="{{ $photo->temporaryUrl() }}"
                                alt="Profile Photo Preview">
                        @else
                            <img class="w-40 h-40 rounded-full border-[6px] border-white object-cover shadow-lg bg-shadedOfGray-20"
                                src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}"
                                alt="Profile Photo">
                        @endif

                        <button type="button" @click="showPhotoModal = true" class="absolute bottom-0 right-0 bg-primary-85 text-white p-2.5 rounded-full border-4 border-white hover:bg-primary-100 transition shadow-md group" title="Change Photo">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                        </button>
                    </div>
                </div>

                <div class="w-full md:w-auto">
                    <button
                        wire:click="update"
                        disabled
                        wire:dirty.attr.remove="disabled"
                        wire:target="fullname,email,dateOfBirth,gender,countryId"
                        class="w-full md:w-auto px-8 py-3 rounded-xl font-medium text-mediumBtn text-white flex items-center justify-center gap-2 transition shadow-lg
                               bg-primary-60 hover:bg-primary-50 shadow-primary-20/50
                               disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-shadedOfGray-30 disabled:shadow-none"
                    >
                        Save Changes
                    </button>
                </div>
            </div>

            @if (session('success'))
                <div class="flex items-start sm:items-center p-4 mb-4 text-sm text-fg-success-strong rounded-base bg-success-soft m-5" role="alert">
                    <svg class="w-4 h-4 me-2 shrink-0 mt-0.5 sm:mt-0" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 11h2v5m-2 0h4m-2.592-8.5h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>
                    <p>{{ session('success')}}</p>
                </div>
            @endif

            <form wire:submit.prevent="update">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-12">
                    {{-- Personal Information Column (Tidak Berubah) --}}
                    <div class="lg:col-span-2 space-y-8">
                        <div>
                            <h2 class="font-primary text-title text-primary-100 mb-2">Personal Information</h2>
                            <p class="text-shadedOfGray-60 text-small mb-8">Manage your personal details and public profile.</p>

                            <div class="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-8">
                                <div class="col-span-1 md:col-span-2">
                                    <label for="fullname" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Full Name</label>
                                    <input type="text" wire:model="fullname" id="fullname" name="fullname" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-white">
                                    @error('fullname') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div class="col-span-1 md:col-span-2">
                                    <label for="email" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Email Address</label>
                                    <input type="email" wire:model="email" id="email" name="email" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white">
                                    @error('email') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div>
                                    <label for="dob" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Date of Birth</label>
                                    <input type="date" wire:model="dateOfBirth" id="dob" name="dob" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white font-secondaryAndButton">
                                    @error('dateOfBirth') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                                <div>
                                    <label for="region" class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Region</label>
                                    <div class="relative">
                                        <select wire:model="countryId" id="region" name="region" class="w-full px-5 py-4 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition appearance-none bg-white text-shadedOfGray-100 cursor-pointer">
                                            @forelse($regions as $region)
                                                <option value="{{ $region->id }}" {{ $countryId === $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                                            @empty @endforelse
                                        </select>
                                        <div class="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none">
                                            <svg class="w-5 h-5 text-shadedOfGray-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-span-1 md:col-span-2">
                                    <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">
                                        Gender
                                    </label>
                                    <div class="flex gap-4">
                                        <label class="cursor-pointer w-full group">
                                            <input type="radio" name="gender" wire:model="gender" value="1" class="peer sr-only">
                                            <div class="w-full p-4 border border-shadedOfGray-20 rounded-xl flex items-center justify-center gap-3 peer-checked:border-primary-60 peer-checked:bg-primary-10/20 peer-checked:text-primary-85 transition group-hover:bg-shadedOfGray-10">
                                                <span class="text-body-size">Male</span>
                                            </div>
                                        </label>
                                        <label class="cursor-pointer w-full group">
                                            <input type="radio" name="gender" wire:model="gender" value="0" class="peer sr-only">
                                            <div class="w-full p-4 border border-shadedOfGray-20 rounded-xl flex items-center justify-center gap-3 peer-checked:border-primary-60 peer-checked:bg-primary-10/20 peer-checked:text-primary-85 transition group-hover:bg-shadedOfGray-10">
                                                <span class="text-body-size">Female</span>
                                            </div>
                                        </label>
                                    </div>
                                    @error('gender') <span class="text-error-moderate">{{ $message }}</span> @enderror
                                </div>
                            </div>
                        </div>
                    </div>

                    {{-- Account Security Column (Updated) --}}
                    <div class="lg:col-span-1">
                        <div class="bg-shadedOfGray-10 rounded-2xl p-8 h-full border border-shadedOfGray-20 flex flex-col justify-between">
                            <div>
                                <div class="flex items-center gap-3 mb-8">
                                    <div class="p-2.5 bg-white border border-shadedOfGray-20 rounded-xl text-primary-100 shadow-sm">
                                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                        </svg>
                                    </div>
                                    <h3 class="font-primary text-body-size font-bold text-shadedOfGray-100">Account Security</h3>
                                </div>

                                <div class="space-y-6">
                                    {{-- Username Section --}}
                                    <div>
                                        <label class="block text-micro font-bold text-shadedOfGray-60 mb-2 uppercase tracking-wider">Username</label>
                                        <div class="relative">
                                            <div class="flex items-center bg-white border border-shadedOfGray-20 rounded-xl px-4 py-3 pr-20">
                                                <span class="text-shadedOfGray-50 mr-2 font-medium">@</span>
                                                <input type="text" wire:model="username" class="w-full outline-none text-body-size text-shadedOfGray-85 font-medium bg-transparent" readonly>
                                            </div>
                                            <button type="button" @click="showUsernameModal = true" class="absolute right-2 top-2 bottom-2 px-3 text-smallBtn text-primary-85 hover:bg-primary-10 rounded-lg font-medium transition">
                                                Edit
                                            </button>
                                        </div>
                                    </div>

                                    {{-- Password Section --}}
                                    <div>
                                        <label class="block text-micro font-bold text-shadedOfGray-60 mb-2 uppercase tracking-wider">Password</label>
                                        <div class="relative">
                                            <input type="password" value="DummyPass123" class="w-full bg-white px-4 py-3 text-body-size border border-shadedOfGray-20 rounded-xl text-shadedOfGray-85 outline-none pr-20" readonly>
                                            <button type="button" @click="showPasswordModal = true" class="absolute right-2 top-2 bottom-2 px-3 text-smallBtn text-accent-100 hover:bg-accent-10 rounded-lg font-medium transition">
                                                Change
                                            </button>
                                        </div>
                                    </div>
                                    <div class="pt-2">
                                        <p class="text-micro text-shadedOfGray-50 italic">Last password changed: 3 months ago</p>
                                    </div>
                                </div>
                            </div>

                            {{-- DELETE ACCOUNT SECTION ADDED --}}
                            <div class="pt-8 mt-8 border-t border-shadedOfGray-20">
                                <h4 class="text-error-moderate font-bold text-smallBtn mb-3">Danger Zone</h4>
                                <button type="button" @click="showDeleteModal = true" class="w-full py-3 border border-error-moderate/30 text-error-moderate bg-error-moderate/5 hover:bg-error-moderate/10 hover:border-error-moderate rounded-xl font-medium transition flex items-center justify-center gap-2">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                    Delete Account
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </form>
        </div>

        {{-- MODAL: PHOTO (Existing) --}}
        <div x-show="showPhotoModal" x-cloak style="display: none;" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100">
                <div class="bg-gradient-to-r from-primary-100 to-primary-60 p-6">
                    <h3 class="font-primary text-subtitle text-white">Update Photo</h3>
                    <p class="text-primary-10 text-small">Select a new image for your profile.</p>
                </div>

                <div class="p-6">
                    @error('profilePhoto')
                    <div class="mb-4 p-3 rounded-lg bg-secondary-angry-10 border border-secondary-angry-20 text-secondary-angry-100 text-smallBtn flex items-start gap-2">
                         <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                        <span>{{ $message }}</span>
                    </div>
                    @enderror

                    <form wire:submit.prevent="uploadPhoto" enctype="multipart/form-data">
                        <div class="space-y-6">
                            <div class="flex justify-center">
                                <div class="relative w-40 h-40">
                                    @if ($photo)
                                        <img src="{{ $photo->temporaryUrl() }}" class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @else
                                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @endif

                                    <label for="photoInputLivewire" class="absolute inset-0 flex items-center justify-center bg-black/40 rounded-full opacity-0 hover:opacity-100 transition cursor-pointer">
                                        <span class="text-white font-medium text-smallBtn">Click to Select</span>
                                    </label>
                                </div>
                            </div>
                            <input type="file" id="photoInputLivewire" wire:model="photo" class="hidden" name="photo">
                            <div class="text-center">
                                <button type="button" onclick="document.getElementById('photoInputLivewire').click()" class="text-primary-50 font-medium text-smallBtn hover:underline">
                                    Choose File
                                </button>
                                <p class="text-micro text-shadedOfGray-50 mt-1">JPG, PNG or GIF. Max 5MB.</p>
                                <div wire:loading wire:target="photo" class="text-secondary-happy-100 text-micro mt-2 font-medium">
                                    Uploading...
                                </div>
                            </div>
                            <div class="flex gap-3 pt-2">
                                <button type="button" @click="showPhotoModal = false" class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-medium hover:bg-shadedOfGray-10 transition">
                                    Cancel
                                </button>
                                <button type="submit" class="flex-1 py-3 bg-primary-60 hover:bg-primary-50 text-white rounded-xl font-medium shadow-md shadow-primary-20/50 transition">
                                    Upload
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {{-- MODAL: CHANGE USERNAME --}}
            <div x-show="showUsernameModal" x-cloak style="display: none;" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
                <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100" @click.away="showUsernameModal = false">
                    <div class="bg-gradient-to-r from-primary-100 to-primary-60 p-6">
                        <h3 class="font-primary text-subtitle text-white">Change Username</h3>
                        <p class="text-primary-10 text-small">Update your unique identifier.</p>
                    </div>
                    <div class="p-6">
                        <form wire:submit.prevent="updateUsername">
                            <div class="space-y-4">
                                <div>
                                    <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">New Username</label>
                                    <div class="flex items-center bg-white border border-shadedOfGray-20 rounded-xl px-4 py-3 focus-within:border-primary-50 focus-within:ring-4 focus-within:ring-primary-10/40 transition">
                                        <span class="text-shadedOfGray-50 mr-2 font-medium">@</span>
                                        <input type="text" wire:model.defer="newUsername" class="w-full outline-none text-body-size text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-transparent" placeholder="new username">
                                    </div>
                                    @error('newUsername') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                                </div>
                                <div>
                                    <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Current Password</label>
                                    <input type="password" wire:model.defer="confirmPassword" class="w-full px-5 py-3 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white" placeholder="Confirm with password">
                                    @error('confirmPassword') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                                </div>
                                <div class="flex gap-3 pt-4">
                                    <button type="button" @click="showUsernameModal = false" class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-medium hover:bg-shadedOfGray-10 transition">
                                        Cancel
                                    </button>
                                    <button type="submit" class="flex-1 py-3 bg-primary-60 hover:bg-primary-50 text-white rounded-xl font-medium shadow-md shadow-primary-20/50 transition">
                                        Update
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>

        {{-- MODAL: CHANGE PASSWORD --}}
        <div x-show="showPasswordModal" x-cloak style="display: none;" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100" @click.away="showPasswordModal = false">
                <div class="bg-gradient-to-r from-accent-100 to-accent-80 p-6">
                    <h3 class="font-primary text-subtitle text-white">Change Password</h3>
                    <p class="text-white/80 text-small">Ensure your account stays secure.</p>
                </div>
                <div class="p-6">
                    <form wire:submit.prevent="updatePassword">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Current Password</label>
                                <input type="password" wire:model.defer="currentPassword" class="w-full px-5 py-3 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white">
                                @error('currentPassword') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                            </div>
                            <div>
                                <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">New Password</label>
                                <input type="password" wire:model.defer="newPassword" class="w-full px-5 py-3 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white">
                                @error('newPassword') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                            </div>
                             <div>
                                <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Confirm New Password</label>
                                <input type="password" wire:model.defer="newPasswordConfirmation" class="w-full px-5 py-3 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white">
                                @error('newPasswordConfirmation') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                            </div>
                            <div class="flex gap-3 pt-4">
                                <button type="button" @click="showPasswordModal = false" class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-medium hover:bg-shadedOfGray-10 transition">
                                    Cancel
                                </button>
                                <button type="submit" class="flex-1 py-3 bg-accent-100 hover:bg-accent-80 text-white rounded-xl font-medium shadow-md shadow-accent-20/50 transition">
                                    Change Password
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {{-- MODAL: DELETE ACCOUNT --}}
        <div x-show="showDeleteModal" x-cloak style="display: none;" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100 border-t-4 border-error-moderate" @click.away="showDeleteModal = false">
                <div class="bg-white p-6 pb-2">
                    <div class="w-12 h-12 rounded-full bg-error-moderate/10 flex items-center justify-center mb-4 mx-auto text-error-moderate">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                    </div>
                    <h3 class="font-primary text-subtitle text-shadedOfGray-100 text-center mb-2">Delete Account</h3>
                    <p class="text-shadedOfGray-60 text-small text-center">Are you sure you want to delete your account? This action is <span class="font-bold text-error-moderate">irreversible</span>. All your data will be permanently removed.</p>
                </div>
                <div class="p-6 pt-2">
                    <form wire:submit.prevent="deleteAccount">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-smallBtn font-medium text-shadedOfGray-70 mb-2">Enter your password to confirm</label>
                                <input type="password" wire:model.defer="deleteConfirmationPassword" class="w-full px-5 py-3 text-body-size border border-shadedOfGray-20 rounded-xl focus:border-error-moderate focus:ring-4 focus:ring-error-moderate/20 outline-none transition text-shadedOfGray-100 bg-white placeholder-shadedOfGray-30">
                                @error('deleteConfirmationPassword') <span class="text-error-moderate text-smallBtn mt-1">{{ $message }}</span> @enderror
                            </div>
                            <div class="flex gap-3 pt-4">
                                <button type="button" @click="showDeleteModal = false" class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-medium hover:bg-shadedOfGray-10 transition">
                                    Cancel
                                </button>
                                <button type="submit" class="flex-1 py-3 bg-error-moderate hover:bg-red-600 text-white rounded-xl font-medium shadow-md shadow-error-moderate/30 transition">
                                    Yes, Delete
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

    </div>
</div>
