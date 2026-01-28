<?php
namespace Tests\Feature\Auth;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class LoginTest extends TestCase
{
    use RefreshDatabase;

    public function test_halaman_login_bisa_diakses()
    {
        $response = $this->get(route('login'));
        $response->assertStatus(200);
        $response->assertViewIs('auth.login');
    }

    public function test_login_berhasil_sebagai_user_biasa()
    {
        // Arrange: Buat user dengan role 0
        $user = User::factory()->create([
            'username' => 'user123',
            'password' => bcrypt('password'),
            'role'     => 0, // User
        ]);

        // Act
        $response = $this->post(route('auth.login'), [
            'username' => 'user123',
            'password' => 'password',
        ]);

        // Assert: Redirect ke user.index
        $response->assertRedirect(route('user.index'));
        $this->assertAuthenticatedAs($user);
    }

    public function test_login_berhasil_sebagai_admin()
    {
        // Arrange: Buat user dengan role 1
        $user = User::factory()->create([
            'username' => 'admin123',
            'password' => bcrypt('password'),
            'role'     => 1, // Admin
        ]);

        // Act
        $response = $this->post(route('auth.login'), [
            'username' => 'admin123',
            'password' => 'password',
        ]);

        // Assert: Redirect ke admin.songs.index
        $response->assertRedirect(route('admin.songs.index'));
    }

    public function test_login_berhasil_sebagai_superadmin()
    {
        // Arrange: Buat user dengan role 2
        $user = User::factory()->create([
            'username' => 'super123',
            'password' => bcrypt('password'),
            'role'     => 2, // Superadmin
        ]);

        // Act
        $response = $this->post(route('auth.login'), [
            'username' => 'super123',
            'password' => 'password',
        ]);

        // Assert: Redirect ke superadmin.users.overview
        $response->assertRedirect(route('superadmin.users.overview'));
    }

    public function test_login_gagal_password_salah()
    {
        $user = User::factory()->create([
            'username' => 'user123',
            'password' => bcrypt('password'),
        ]);

        $response = $this->from(route('login'))->post(route('auth.login'), [
            'username' => 'user123',
            'password' => 'salah',
        ]);

        $response->assertRedirect(route('login'));
        $response->assertSessionHas('failed', 'username or password incorrect!');
        $this->assertGuest(); // Pastikan tidak login
    }

    public function test_login_gagal_validasi_kosong()
    {
        $response = $this->post(route('auth.login'), []);
        $response->assertSessionHasErrors(['username', 'password']);
    }
}
