<?php
namespace App\Services;

use DateInterval;
use DateTime;
use MicrosoftAzure\Storage\Blob\BlobSharedAccessSignatureHelper;

class AzureSasService
{
    protected BlobSharedAccessSignatureHelper $helper;
    protected string $account;
    protected string $container;

    public function __construct()
    {
        $this->account   = config('filesystems.disks.azure.name');
        $this->container = config('filesystems.disks.azure.container');

        $this->helper = new BlobSharedAccessSignatureHelper(
            $this->account,
            config('filesystems.disks.azure.key')
        );
    }

    public function blob(string $blobName, int $minutes = 60): string
    {
        // 1️⃣ expiry HARUS string
        $expiry = (new DateTime())
            ->add(new DateInterval("PT{$minutes}M"))
            ->format('Y-m-d\TH:i:s\Z');

        // 2️⃣ canonical resource HARUS seperti ini
        $canonicalizedResource = sprintf(
            '/blob/%s/%s/%s',
            $this->account,
            $this->container,
            $blobName
        );

        // 3️⃣ URUTAN PARAMETER FIX
        $sas = $this->helper
            ->generateBlobServiceSharedAccessSignatureToken(
                'b',                   // ✅ signedResource (blob)
                'r',                   // ✅ signedPermissions (INI YANG DICEK ERROR)
                $expiry,               // expiry
                $canonicalizedResource // canonical resource
            );

        return sprintf(
            'https://%s.blob.core.windows.net/%s/%s?%s',
            $this->account,
            $this->container,
            $blobName,
            $sas
        );
    }
}
